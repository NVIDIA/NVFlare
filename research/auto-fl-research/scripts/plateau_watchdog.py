#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recommend whether an Auto-FL campaign should switch to literature review."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_SCORED_SINCE_RESET = 32
DEFAULT_MIN_DELTA = 0.0005
NON_PLATEAU_STATUSES = {"crash", "literature"}
ASSIGNMENT_PATTERN = re.compile(r"\b([A-Za-z][A-Za-z0-9_+-]*)=([^\s,;\]\)]+)")
WORD_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_+-]{2,}\b")
STOPWORDS = {
    "and",
    "audit",
    "candidate",
    "control",
    "discard",
    "extra",
    "keep",
    "kept",
    "precision",
    "rerun",
    "repro",
    "run",
    "sweep",
    "with",
}


@dataclass
class ResultRow:
    index: int
    commit: str
    score: float | None
    runtime_seconds: float
    status: str
    target: str
    description: str


def parse_score(value: str) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score) or math.isinf(score):
        return None
    return score


def parse_runtime(value: str) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(seconds) or math.isinf(seconds):
        return 0.0
    return max(0.0, seconds)


def load_results(path: Path) -> list[ResultRow]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = []
        for index, row in enumerate(reader):
            rows.append(
                ResultRow(
                    index=index,
                    commit=row.get("commit", ""),
                    score=parse_score(row.get("score", "")),
                    runtime_seconds=parse_runtime(row.get("runtime_seconds", "")),
                    status=(row.get("status", "") or "").strip().lower(),
                    target=row.get("target", ""),
                    description=" ".join((row.get("description", "") or "").split()),
                )
            )
    return rows


def is_scored_candidate(row: ResultRow) -> bool:
    return row.score is not None and row.status not in NON_PLATEAU_STATUSES


def last_material_improvement(rows: list[ResultRow], min_delta: float) -> ResultRow | None:
    material_best = None
    last_improvement = None
    for row in rows:
        if row.score is None:
            continue
        if material_best is None or row.score > material_best + min_delta:
            material_best = row.score
            last_improvement = row
    return last_improvement


def actual_best(rows: list[ResultRow]) -> ResultRow | None:
    scored_rows = [row for row in rows if row.score is not None]
    if not scored_rows:
        return None
    return max(scored_rows, key=lambda row: row.score)


def last_literature(rows: list[ResultRow]) -> ResultRow | None:
    literature_rows = [row for row in rows if row.status == "literature"]
    if not literature_rows:
        return None
    return literature_rows[-1]


def reset_anchor(
    material_row: ResultRow | None,
    literature_row: ResultRow | None,
) -> tuple[str, ResultRow | None]:
    if material_row is None and literature_row is None:
        return "start", None
    if literature_row is not None and (material_row is None or literature_row.index > material_row.index):
        return "literature", literature_row
    return "material_improvement", material_row


def rows_after_anchor(rows: list[ResultRow], anchor: ResultRow | None) -> list[ResultRow]:
    if anchor is None:
        return rows
    return [row for row in rows if row.index > anchor.index]


def normalize_assignment_value(value: str) -> str:
    return value.strip().strip("`'\".,:;")


def repeated_description_terms(rows: list[ResultRow], max_terms: int, min_count: int) -> str:
    counter: Counter[str] = Counter()
    for row in rows:
        description = row.description
        assignment_spans = []
        for match in ASSIGNMENT_PATTERN.finditer(description):
            key = match.group(1).lower()
            value = normalize_assignment_value(match.group(2).lower())
            if value:
                counter[f"{key}={value}"] += 1
            counter[f"{key}=*"] += 1
            assignment_spans.append(match.span())

        masked = list(description)
        for start, end in assignment_spans:
            for index in range(start, end):
                masked[index] = " "
        for word in WORD_PATTERN.findall("".join(masked).lower()):
            if word not in STOPWORDS:
                counter[word] += 1

    selected = []
    for term, count in counter.most_common():
        if count < min_count:
            continue
        if term.endswith("=*"):
            prefix = term[:-1]
            if any(existing.startswith(prefix) and existing_count == count for existing, existing_count in selected):
                continue
        selected.append((term, count))
        if len(selected) >= max_terms:
            break

    if not selected:
        return ""
    return ", ".join(f"{term}:{count}" for term, count in selected)


def format_score(row: ResultRow | None) -> str:
    if row is None or row.score is None:
        return ""
    return f"{row.score:.6f}"


def format_index(row: ResultRow | None) -> str:
    if row is None:
        return ""
    return str(row.index)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="results.tsv")
    parser.add_argument(
        "--max-scored-since-reset",
        type=int,
        default=DEFAULT_MAX_SCORED_SINCE_RESET,
        help="Recommend literature mode after this many scored non-crash rows since the last reset.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=DEFAULT_MIN_DELTA,
        help="Minimum score gain required to reset the plateau clock.",
    )
    parser.add_argument("--max-terms", type=int, default=6, help="Maximum repeated description terms to print.")
    parser.add_argument("--min-term-count", type=int, default=2, help="Minimum repeated-term count to print.")
    args = parser.parse_args()

    if args.max_scored_since_reset <= 0:
        raise ValueError("--max-scored-since-reset must be positive")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be non-negative")

    path = Path(args.path)
    rows = load_results(path)
    best_row = actual_best(rows)
    material_row = last_material_improvement(rows, args.min_delta)
    literature_row = last_literature(rows)
    reset_source, anchor = reset_anchor(material_row, literature_row)
    recent_rows = rows_after_anchor(rows, anchor)
    scored_since_reset = [row for row in recent_rows if is_scored_candidate(row)]
    recommendation = "literature" if len(scored_since_reset) >= args.max_scored_since_reset else "continue"

    print(f"recommendation={recommendation}")
    if recommendation == "literature":
        print(
            "reason=scored candidates since the last material improvement or literature reset reached "
            f"{len(scored_since_reset)}/{args.max_scored_since_reset}"
        )
    else:
        print(
            "reason=scored candidates since the last material improvement or literature reset are below "
            f"{len(scored_since_reset)}/{args.max_scored_since_reset}"
        )
    print(f"scored_since_reset={len(scored_since_reset)}")
    print(f"threshold={args.max_scored_since_reset}")
    print(f"min_delta={args.min_delta:.6f}")
    print(f"reset_source={reset_source}")
    print(f"reset_index={format_index(anchor)}")
    print(f"best_index={format_index(best_row)}")
    print(f"best_score={format_score(best_row)}")
    print(f"last_material_improvement_index={format_index(material_row)}")
    print(f"last_material_improvement_score={format_score(material_row)}")
    print(f"last_literature_index={format_index(literature_row)}")
    if literature_row is not None:
        print(f"last_literature_runtime_seconds={literature_row.runtime_seconds:.0f}")
    terms = repeated_description_terms(scored_since_reset, args.max_terms, args.min_term_count)
    if terms:
        print(f"repeated_terms={terms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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

"""Promote or discard reviewed candidate rows in results.tsv."""

import argparse
import csv
import math
from pathlib import Path

ALLOWED_STATUSES = {"candidate", "keep", "discard", "crash"}


def parse_score(row):
    try:
        score = float(row.get("score", "") or 0.0)
    except ValueError:
        return None
    if math.isnan(score) or math.isinf(score):
        return None
    return score


def load_rows(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise ValueError(f"{path} has no TSV header")
    if "status" not in fieldnames or "score" not in fieldnames:
        raise ValueError(f"{path} must contain status and score columns")
    return fieldnames, rows


def write_rows(path, fieldnames, rows):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def selected_indices(rows, args):
    if args.all_candidates:
        return {index for index, row in enumerate(rows) if row.get("status", "").strip().lower() == "candidate"}
    start = max(0, len(rows) - args.last)
    return set(range(start, len(rows)))


def previous_best(rows, selected):
    scores = [
        parse_score(row)
        for index, row in enumerate(rows)
        if index not in selected and row.get("status", "").strip().lower() != "crash"
    ]
    scores = [score for score in scores if score is not None]
    return max(scores) if scores else None


def choose_best_candidate(rows, selected):
    candidate_indices = [
        index
        for index in selected
        if rows[index].get("status", "").strip().lower() == "candidate" and parse_score(rows[index]) is not None
    ]
    if not candidate_indices:
        return None
    return max(candidate_indices, key=lambda index: parse_score(rows[index]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="results.tsv")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--last", type=int, help="Finalize the last N ledger rows.")
    group.add_argument(
        "--all-candidates",
        action="store_true",
        help="Finalize all current candidate rows. Useful for cleaning stale ledgers.",
    )
    parser.add_argument(
        "--keep-best",
        action="store_true",
        help="Mark the best selected candidate as keep if it improves over prior rows.",
    )
    parser.add_argument(
        "--force-keep-best",
        action="store_true",
        help="Mark the best selected candidate as keep even without an observed improvement.",
    )
    parser.add_argument(
        "--discard-others",
        action="store_true",
        help="Mark selected candidate rows that were not kept as discard.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum absolute improvement required for --keep-best. Default: 0.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.last is not None and args.last <= 0:
        raise ValueError("--last must be positive")
    if not args.keep_best and not args.force_keep_best and not args.discard_others:
        raise ValueError("Select at least one action: --keep-best, --force-keep-best, or --discard-others")

    path = Path(args.path)
    fieldnames, rows = load_rows(path)
    selected = selected_indices(rows, args)
    if not selected:
        print("No rows selected.")
        return 0

    best_index = None
    best_score = None
    prior_best = previous_best(rows, selected)
    if args.keep_best or args.force_keep_best:
        candidate_best = choose_best_candidate(rows, selected)
        if candidate_best is not None:
            candidate_score = parse_score(rows[candidate_best])
            improved = prior_best is None or candidate_score > prior_best + args.min_delta
            if args.force_keep_best or improved:
                best_index = candidate_best
                best_score = candidate_score

    changes = []
    for index in sorted(selected):
        old_status = rows[index].get("status", "").strip().lower()
        if old_status not in ALLOWED_STATUSES:
            continue
        new_status = old_status
        if index == best_index:
            new_status = "keep"
        elif args.discard_others and old_status == "candidate":
            new_status = "discard"
        if new_status != old_status:
            changes.append((index, old_status, new_status, rows[index].get("description", "")))
            rows[index]["status"] = new_status

    if args.dry_run:
        print(f"Dry run: {len(changes)} status changes")
    else:
        write_rows(path, fieldnames, rows)
        print(f"Updated {path}: {len(changes)} status changes")
    if prior_best is not None:
        print(f"previous_best={prior_best:.6f}")
    if best_index is not None:
        print(f"kept=#{best_index} score={best_score:.6f} {rows[best_index].get('description', '')}")
    else:
        print("kept=none")
    for index, old_status, new_status, description in changes[:20]:
        print(f"#{index}: {old_status} -> {new_status}: {description}")
    if len(changes) > 20:
        print(f"... {len(changes) - 20} more changes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

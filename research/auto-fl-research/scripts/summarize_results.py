#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Summarize and rank Auto-FL result ledger rows."""

import argparse
import csv
from pathlib import Path


def parse_score(row):
    try:
        return float(row.get("score", "0") or 0.0)
    except ValueError:
        return 0.0


def parse_runtime(row):
    try:
        return float(row.get("runtime_seconds", "") or 0.0)
    except ValueError:
        return 0.0


def format_runtime(seconds):
    if seconds <= 0:
        return ""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def rank_key(row):
    runtime = parse_runtime(row)
    runtime_tiebreak = runtime if runtime > 0 else float("inf")
    return (-parse_score(row), runtime_tiebreak)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="results.tsv")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--budget", default=None)
    parser.add_argument("--status", default=None)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"results ledger not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    if args.budget is not None:
        rows = [row for row in rows if row.get("budget") == args.budget]
    if args.status is not None:
        rows = [row for row in rows if row.get("status") == args.status]

    selected_rows = rows
    selected_runtime = sum(parse_runtime(row) for row in selected_rows)

    rows = sorted(selected_rows, key=rank_key)[: args.top]
    displayed_runtime = sum(parse_runtime(row) for row in rows)
    if selected_runtime > 0:
        print(f"Total runtime for selected rows: {format_runtime(selected_runtime)}")
        if displayed_runtime != selected_runtime:
            print(f"Runtime for displayed rows: {format_runtime(displayed_runtime)}")
        print("Ranking: score first; runtime is an exact-score tie-break and coarse cost signal.")

    print("| rank | score | runtime | status | target | description | artifacts |")
    print("| ---: | ---: | ---: | --- | --- | --- | --- |")
    for rank, row in enumerate(rows, start=1):
        print(
            "| {rank} | {score:.6f} | {runtime} | {status} | {target} | {description} | {artifacts} |".format(
                rank=rank,
                score=parse_score(row),
                runtime=format_runtime(parse_runtime(row)),
                status=row.get("status", ""),
                target=row.get("target", ""),
                description=row.get("description", ""),
                artifacts=row.get("artifacts", ""),
            )
        )

    if args.status == "candidate" and rows:
        print()
        print(
            "After reviewing a completed run or batch, finalize statuses with "
            "`scripts/finalize_batch_status.py` so kept/discarded runs are visible in progress plots."
        )


if __name__ == "__main__":
    main()

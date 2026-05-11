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

"""Append one safely quoted row to the Auto-FL results.tsv ledger."""

import argparse
import csv
import os
from pathlib import Path

FIELDS = ["commit", "score", "runtime_seconds", "budget", "status", "target", "description", "artifacts"]
OLD_FIELDS = ["commit", "score", "budget", "status", "target", "description", "artifacts"]


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        return reader.fieldnames, rows


def write_rows(path: Path, rows):
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def ensure_results_tsv(path: Path):
    if not path.exists():
        write_rows(path, [])
        return

    fieldnames, rows = read_rows(path)
    if fieldnames == FIELDS:
        return

    if fieldnames == OLD_FIELDS:
        migrated_rows = []
        for row in rows:
            migrated_rows.append(
                {
                    "commit": row.get("commit", ""),
                    "score": row.get("score", ""),
                    "runtime_seconds": "",
                    "budget": row.get("budget", ""),
                    "status": row.get("status", ""),
                    "target": row.get("target", ""),
                    "description": row.get("description", ""),
                    "artifacts": row.get("artifacts", ""),
                }
            )
        write_rows(path, migrated_rows)
        return

    raise ValueError(f"Unknown {path} header: {fieldnames}")


def append_result(path: Path, row):
    if path.parent != Path(""):
        path.parent.mkdir(parents=True, exist_ok=True)
    ensure_results_tsv(path)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.tsv")
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Create or migrate the results.tsv header without appending a row.",
    )
    parser.add_argument("--commit")
    parser.add_argument("--score", default="")
    parser.add_argument("--runtime-seconds")
    parser.add_argument("--budget")
    parser.add_argument("--status", choices=["candidate", "crash", "discard", "keep", "literature"])
    parser.add_argument("--target")
    parser.add_argument("--description")
    parser.add_argument("--artifacts")
    args = parser.parse_args()
    if args.init_only:
        path = Path(args.results)
        if path.parent != Path(""):
            path.parent.mkdir(parents=True, exist_ok=True)
        ensure_results_tsv(path)
        print(f"Ensured {args.results}")
        return

    missing = [
        name
        for name in ["commit", "runtime_seconds", "budget", "status", "target", "description", "artifacts"]
        if getattr(args, name) is None
    ]
    if missing:
        parser.error("missing required arguments: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    if args.status != "literature" and args.score == "":
        parser.error("--score is required unless --status=literature")

    append_result(
        Path(args.results),
        {
            "commit": args.commit,
            "score": args.score,
            "runtime_seconds": args.runtime_seconds,
            "budget": args.budget,
            "status": args.status,
            "target": args.target,
            "description": args.description,
            "artifacts": args.artifacts,
        },
    )


if __name__ == "__main__":
    main()

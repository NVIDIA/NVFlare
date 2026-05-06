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

"""Record literature-review cycles in the Auto-FL results.tsv ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from append_result import append_result

DEFAULT_TARGET = "templates/literature_loop.md"
DEFAULT_BUDGET = "literature_loop"


def compact_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip() or "no-git"
    except Exception:
        return "no-git"


def default_timer_path(results_path: Path) -> Path:
    digest = hashlib.sha1(str(results_path.resolve().parent).encode("utf-8")).hexdigest()[:12]
    tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
    return tmp_root / f"autofl_literature_review_{digest}.json"


def read_timer(path: Path) -> dict[str, object] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def write_timer(path: Path, results_path: Path, description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "started_at": time.time(),
        "results": str(results_path),
        "description": description,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
        f.write("\n")


def append_literature_row(args, runtime_seconds: float, description: str) -> None:
    description = compact_text(description)
    if not description:
        description = "literature review"
    append_result(
        Path(args.results),
        {
            "commit": git_commit(),
            "score": "",
            "runtime_seconds": f"{max(0.0, runtime_seconds):.0f}",
            "budget": args.budget,
            "status": "literature",
            "target": args.target,
            "description": description,
            "artifacts": args.artifacts,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results.tsv")
    parser.add_argument("--description", default="")
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--artifacts", default=DEFAULT_TARGET)
    parser.add_argument("--budget", default=DEFAULT_BUDGET)
    parser.add_argument("--runtime-seconds", type=float, default=None)
    parser.add_argument("--timer-path", default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--start", action="store_true", help="Start timing a literature-review cycle.")
    mode.add_argument("--finish", action="store_true", help="Append a literature row using the active timer.")
    mode.add_argument("--log", action="store_true", help="Append a literature row immediately.")
    args = parser.parse_args()

    results_path = Path(args.results)
    timer_path = Path(args.timer_path) if args.timer_path else default_timer_path(results_path)

    if args.start:
        write_timer(timer_path, results_path, compact_text(args.description))
        print(f"Started literature-review timer: {timer_path}")
        return 0

    description = compact_text(args.description)
    runtime_seconds = args.runtime_seconds
    timer = None
    if args.finish:
        timer = read_timer(timer_path)
        if timer is not None and runtime_seconds is None:
            runtime_seconds = time.time() - float(timer.get("started_at", time.time()))
        if timer is not None and not description:
            description = compact_text(str(timer.get("description", "")))
        if timer is None and runtime_seconds is None:
            print(
                f"No active literature-review timer at {timer_path}; logging with runtime_seconds=0.",
                file=sys.stderr,
            )
            runtime_seconds = 0.0
    elif runtime_seconds is None:
        runtime_seconds = 0.0

    append_literature_row(args, runtime_seconds or 0.0, description)
    if args.finish and timer is not None:
        timer_path.unlink(missing_ok=True)
    print(f"Appended literature-review event to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

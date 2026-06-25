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

"""Write the deterministic Auto-FL campaign continuation state.

This script owns the stop/continue decision for the research harness. Agents may
choose mutations, but they must not decide that a campaign is complete while this
guard says another comparable batch should run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MAX_SCORED_SINCE_RESET = 32
DEFAULT_MIN_DELTA = 0.0005
DEFAULT_STATE_PATH = ".autoresearch/campaign_state.json"
DEFAULT_STOP_FILES = ("STOP_AUTOFL", ".autoresearch/STOP_AUTOFL", ".nvflare/autofl/STOP")
COMPARABLE_STATUSES = {"candidate", "keep", "discard", "crash"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_score(value: str) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score) or math.isinf(score):
        return None
    return score


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def normalize_status(row: dict[str, str]) -> str:
    return (row.get("status", "") or "").strip().lower()


def is_baseline(row: dict[str, str]) -> bool:
    text = " ".join(
        [
            row.get("description", "") or "",
            row.get("budget", "") or "",
            row.get("artifacts", "") or "",
        ]
    ).lower()
    return "baseline" in text or "--name baseline" in text


def comparable_attempts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if normalize_status(row) in COMPARABLE_STATUSES and not is_baseline(row)]


def scored_attempts(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in comparable_attempts(rows) if parse_score(row.get("score", "")) is not None]


def pending_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if normalize_status(row) == "candidate"]


def best_score(rows: list[dict[str, str]]) -> float | None:
    scores = [parse_score(row.get("score", "")) for row in rows]
    scores = [score for score in scores if score is not None]
    if not scores:
        return None
    return max(scores)


def parse_max_candidates(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed


def parse_max_candidates_arg(value: str) -> int:
    parsed = parse_max_candidates(value)
    if parsed is None:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def existing_stop_files(paths: list[str]) -> list[str]:
    return [path for path in paths if Path(path).exists()]


def repeated_crash_blocker(attempts: list[dict[str, str]], threshold: int) -> bool:
    if threshold <= 0 or len(attempts) < threshold:
        return False
    return all(normalize_status(row) == "crash" for row in attempts[-threshold:])


def parse_key_value_output(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def run_watchdog(results_path: Path, threshold: int, min_delta: float) -> dict[str, Any]:
    watchdog = Path(__file__).resolve().with_name("plateau_watchdog.py")
    if not watchdog.exists() or not results_path.exists():
        return {"available": False, "recommendation": "continue", "raw": ""}

    command = [
        sys.executable,
        str(watchdog),
        str(results_path),
        "--max-scored-since-reset",
        str(threshold),
        "--min-delta",
        str(min_delta),
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=False)
    raw = process.stdout.strip()
    parsed = parse_key_value_output(raw)
    recommendation = parsed.get("recommendation") or "continue"
    return {
        "available": True,
        "returncode": process.returncode,
        "recommendation": recommendation,
        "fields": parsed,
        "raw": raw,
        "error": process.stderr.strip(),
    }


def guard_state(args) -> dict[str, Any]:
    results_path = Path(args.results)
    rows = load_rows(results_path)
    attempts = comparable_attempts(rows)
    pending = pending_candidates(rows)
    cap = args.max_candidates
    if cap is None:
        cap = parse_max_candidates(os.environ.get("AUTOFL_MAX_CANDIDATES"))
    stop_files = existing_stop_files(args.stop_file)
    watchdog = run_watchdog(results_path, args.plateau_threshold, args.min_delta)

    decision = "continue"
    reason = "continue"
    next_action = "launch_next_candidate_batch"
    final_response_allowed = False

    if pending:
        reason = "pending_candidates"
        next_action = "finalize_pending_candidates"
    elif stop_files:
        decision = "stop"
        reason = "manual_stop_file"
        next_action = "final_report"
        final_response_allowed = True
    elif cap is not None and len(attempts) >= cap:
        decision = "stop"
        reason = "candidate_cap_exhausted"
        next_action = "final_report"
        final_response_allowed = True
    elif repeated_crash_blocker(attempts, args.hard_crash_threshold):
        decision = "stop"
        reason = "hard_repeated_crash_blocker"
        next_action = "final_report"
        final_response_allowed = True
    elif watchdog.get("recommendation") == "literature":
        reason = "plateau_literature"
        next_action = "run_literature_loop"

    if final_response_allowed:
        instruction = "Final report is allowed because the campaign guard reached a stop condition."
    elif next_action == "finalize_pending_candidates":
        instruction = (
            "Do not produce a final answer. Finalize reviewed candidate rows, refresh artifacts, then rerun the guard."
        )
    elif next_action == "run_literature_loop":
        instruction = "Do not produce a final answer. Run the literature loop, log the event, and launch source-backed candidates."
    else:
        instruction = "Do not produce a final answer. Launch the next same-budget candidate batch now."

    return {
        "schema_version": "nvflare.autofl.campaign_state.v1",
        "updated_at": utc_now(),
        "results": str(results_path),
        "decision": decision,
        "reason": reason,
        "next_action": next_action,
        "final_response_allowed": final_response_allowed,
        "candidate_cap": cap,
        "candidate_attempts": len(attempts),
        "pending_candidates": len(pending),
        "scored_attempts": len(scored_attempts(rows)),
        "best_score": best_score(rows),
        "stop_files": stop_files,
        "watchdog": watchdog,
        "agent_instruction": instruction,
    }


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def print_text(state: dict[str, Any]) -> None:
    for key in [
        "decision",
        "reason",
        "next_action",
        "final_response_allowed",
        "candidate_cap",
        "candidate_attempts",
        "pending_candidates",
        "scored_attempts",
        "best_score",
        "agent_instruction",
    ]:
        value = state.get(key)
        if isinstance(value, bool):
            value = str(value).lower()
        elif value is None:
            value = ""
        print(f"{key}={value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="?", default="results.tsv")
    parser.add_argument("--state", default=DEFAULT_STATE_PATH)
    parser.add_argument("--max-candidates", type=parse_max_candidates_arg)
    parser.add_argument("--stop-file", action="append", default=list(DEFAULT_STOP_FILES))
    parser.add_argument("--plateau-threshold", type=int, default=DEFAULT_MAX_SCORED_SINCE_RESET)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--hard-crash-threshold", type=int, default=6)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    if args.plateau_threshold <= 0:
        raise ValueError("--plateau-threshold must be positive")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be non-negative")

    state = guard_state(args)
    write_state(Path(args.state), state)
    if args.format == "json":
        print(json.dumps(state, sort_keys=True))
    else:
        print_text(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

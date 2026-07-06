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

"""Diagnose the product Auto-FL campaign continuation decision.

The skill runner executes candidates and writes ``results.tsv``. This guard
turns that ledger into a machine-readable decision. The campaign runner is the
only writer of authoritative campaign state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_HARD_CRASH_THRESHOLD = 6
DEFAULT_MIN_DELTA = 0.0005
DEFAULT_PLATEAU_THRESHOLD = 8
DEFAULT_STOP_FILES = ("STOP_AUTOFL", ".nvflare/autofl/STOP")
ATTEMPT_STATUSES = {"candidate", "keep", "discard", "crash"}
SCORED_ATTEMPT_STATUSES = {"keep", "discard"}
LITERATURE_EVENT_STATUSES = {"event", "literature", "checkpoint"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(score) or math.isinf(score):
        return None
    return score


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def normalize_status(row: Dict[str, str]) -> str:
    return (row.get("status", "") or "").strip().lower()


def row_text(row: Dict[str, str]) -> str:
    return " ".join(str(value or "") for value in row.values()).lower()


def is_baseline(row: Dict[str, str]) -> bool:
    return normalize_status(row) == "baseline"


def is_literature_event(row: Dict[str, str]) -> bool:
    status = normalize_status(row)
    if status in LITERATURE_EVENT_STATUSES and "literature" in row_text(row):
        return True
    return "literature" in (row.get("name", "") or "").lower()


def comparable_attempts(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [row for row in rows if normalize_status(row) in ATTEMPT_STATUSES and not is_baseline(row)]


def pending_candidates(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [row for row in rows if normalize_status(row) == "candidate"]


def scored_attempts_with_index(rows: List[Dict[str, str]]) -> List[Tuple[int, Dict[str, str], float]]:
    scored = []
    for idx, row in enumerate(rows):
        if normalize_status(row) not in SCORED_ATTEMPT_STATUSES or is_baseline(row):
            continue
        score = parse_score(row.get("score", ""))
        if score is not None:
            scored.append((idx, row, score))
    return scored


def better(new_score: float, old_score: Optional[float], mode: str, min_delta: float = 0.0) -> bool:
    if old_score is None:
        return True
    if mode == "min":
        return new_score < old_score - min_delta
    return new_score > old_score + min_delta


def best_score(rows: List[Dict[str, str]], mode: str) -> Optional[float]:
    best = None
    for row in rows:
        status = normalize_status(row)
        retained = status == "keep" or (is_baseline(row) and status not in ATTEMPT_STATUSES)
        if not retained:
            continue
        score = parse_score(row.get("score", ""))
        if score is None:
            continue
        if better(score, best, mode):
            best = score
    return best


def plateau_status(rows: List[Dict[str, str]], threshold: int, min_delta: float, mode: str) -> Dict[str, Any]:
    retained = []
    for idx, row in enumerate(rows):
        status = normalize_status(row)
        is_retained = status == "keep" or (is_baseline(row) and status not in ATTEMPT_STATUSES)
        if not is_retained:
            continue
        score = parse_score(row.get("score", ""))
        if score is not None:
            retained.append((idx, row, score))
    scored_attempts = scored_attempts_with_index(rows)
    if threshold <= 0 or not retained:
        return {
            "available": True,
            "recommendation": "continue",
            "scored_since_reset": 0,
            "threshold": threshold,
            "min_delta": min_delta,
        }

    best = None
    best_row_idx = -1
    best_scored_idx = -1
    best_name = ""
    for scored_idx, (row_idx, row, score) in enumerate(retained):
        if better(score, best, mode, min_delta):
            best = score
            best_row_idx = row_idx
            best_scored_idx = scored_idx
            best_name = row.get("name", "")

    last_literature_idx = max((idx for idx, row in enumerate(rows) if is_literature_event(row)), default=-1)
    reset_row_idx = max(best_row_idx, last_literature_idx)
    scored_since_reset = sum(1 for row_idx, _, _ in scored_attempts if row_idx > reset_row_idx)
    recommendation = "literature" if scored_since_reset >= threshold else "continue"
    return {
        "available": True,
        "recommendation": recommendation,
        "best_name": best_name,
        "best_score": best,
        "best_scored_index": best_scored_idx,
        "last_literature_event_index": last_literature_idx,
        "min_delta": min_delta,
        "reset_row_index": reset_row_idx,
        "scored_since_reset": scored_since_reset,
        "threshold": threshold,
    }


def parse_max_candidates(value: Optional[str]) -> Optional[int]:
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


def existing_stop_files(paths: List[str]) -> List[str]:
    return [path for path in paths if Path(path).exists()]


def repeated_crash_blocker(attempts: List[Dict[str, str]], threshold: int) -> bool:
    if threshold <= 0 or len(attempts) < threshold:
        return False
    return all(normalize_status(row) == "crash" for row in attempts[-threshold:])


def guard_state_for_rows(
    rows: List[Dict[str, str]],
    *,
    results_path: str = "results.tsv",
    max_candidates: Optional[int] = None,
    stop_files: Optional[List[str]] = None,
    plateau_threshold: int = DEFAULT_PLATEAU_THRESHOLD,
    min_delta: float = DEFAULT_MIN_DELTA,
    hard_crash_threshold: int = DEFAULT_HARD_CRASH_THRESHOLD,
    mode: str = "max",
    pending_manifest_count: int = 0,
) -> Dict[str, Any]:
    attempts = comparable_attempts(rows)
    pending = pending_candidates(rows)
    cap = max_candidates
    cap_source = "explicit" if cap is not None else "uncapped"
    stop_file_hits = existing_stop_files(stop_files or list(DEFAULT_STOP_FILES))
    plateau = plateau_status(rows, plateau_threshold, min_delta, mode)

    decision = "continue"
    reason = "continue"
    next_action = "propose_candidate"
    final_response_allowed = False

    pending_count = len(pending) + pending_manifest_count
    if stop_file_hits:
        decision = "stop"
        if pending_count:
            reason = "manual_stop_pending_candidate"
            next_action = "abandon_candidate"
        else:
            reason = "manual_stop_file"
            next_action = "final_report"
            final_response_allowed = True
    elif pending_count:
        reason = "pending_candidates"
        next_action = "edit_candidate"
    elif cap is not None and len(attempts) >= cap:
        decision = "stop"
        reason = "candidate_cap_exhausted"
        next_action = "final_report"
        final_response_allowed = True
    elif repeated_crash_blocker(attempts, hard_crash_threshold):
        decision = "stop"
        reason = "hard_repeated_crash_blocker"
        next_action = "final_report"
        final_response_allowed = True
    elif plateau.get("recommendation") == "literature":
        reason = "plateau_literature"
        next_action = "run_literature_loop"

    if final_response_allowed:
        instruction = "Final report is allowed because authoritative campaign state reached a stop condition."
    elif next_action == "abandon_candidate":
        instruction = (
            "Manual stop requested. Do not execute the pending candidate. Run the runner abandon action, "
            "then refresh status and generate the final report."
        )
    elif next_action == "edit_candidate":
        instruction = "Do not produce a final answer. Finish the pending candidate, then run the runner status action."
    elif next_action == "run_literature_loop":
        instruction = (
            "Do not produce a final answer. Run the literature loop, record a literature event, "
            "then launch source-backed candidates under the same comparison budget. Include at least one "
            "server aggregation candidate when compatible with the job contract; otherwise record why it is incompatible."
        )
    else:
        instruction = "Do not produce a final answer. Propose and prepare the next same-budget candidate now."

    return {
        "schema_version": "nvflare.autofl.campaign_state.v1",
        "updated_at": utc_now(),
        "results": results_path,
        "decision": decision,
        "reason": reason,
        "next_action": next_action,
        "final_response_allowed": final_response_allowed,
        "candidate_cap": cap,
        "candidate_cap_source": cap_source,
        "candidate_attempts": len(attempts),
        "pending_candidates": pending_count,
        "scored_attempts": len(scored_attempts_with_index(rows)),
        "best_score": best_score(rows, mode),
        "stop_files": stop_file_hits,
        "plateau": plateau,
        "required_exploration": "source_backed_server_aggregation" if next_action == "run_literature_loop" else None,
        "agent_instruction": instruction,
    }


def guard_state(
    results_path: Path,
    *,
    max_candidates: Optional[int] = None,
    stop_files: Optional[List[str]] = None,
    plateau_threshold: int = DEFAULT_PLATEAU_THRESHOLD,
    min_delta: float = DEFAULT_MIN_DELTA,
    hard_crash_threshold: int = DEFAULT_HARD_CRASH_THRESHOLD,
    mode: str = "max",
    pending_manifest_count: int = 0,
) -> Dict[str, Any]:
    return guard_state_for_rows(
        load_rows(results_path),
        results_path=str(results_path),
        max_candidates=max_candidates,
        stop_files=stop_files,
        plateau_threshold=plateau_threshold,
        min_delta=min_delta,
        hard_crash_threshold=hard_crash_threshold,
        mode=mode,
        pending_manifest_count=pending_manifest_count,
    )


def print_text(state: Dict[str, Any]) -> None:
    for key in [
        "decision",
        "reason",
        "next_action",
        "final_response_allowed",
        "candidate_cap",
        "candidate_cap_source",
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", default="results.tsv")
    parser.add_argument("--max-candidates", type=parse_max_candidates_arg)
    parser.add_argument("--stop-file", action="append")
    parser.add_argument("--plateau-threshold", type=int, default=DEFAULT_PLATEAU_THRESHOLD)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--hard-crash-threshold", type=int, default=DEFAULT_HARD_CRASH_THRESHOLD)
    parser.add_argument("--mode", choices=["max", "min"], default="max")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)

    if args.plateau_threshold <= 0:
        raise ValueError("--plateau-threshold must be positive")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be non-negative")

    results_path = Path(args.results).resolve()
    stop_files = args.stop_file if args.stop_file is not None else list(DEFAULT_STOP_FILES)
    resolved_stop_files = [
        str(path if path.is_absolute() else results_path.parent / path) for path in map(Path, stop_files)
    ]
    state = guard_state(
        results_path,
        max_candidates=args.max_candidates,
        stop_files=resolved_stop_files,
        plateau_threshold=args.plateau_threshold,
        min_delta=args.min_delta,
        hard_crash_threshold=args.hard_crash_threshold,
        mode=args.mode,
    )
    if args.format == "json":
        print(json.dumps(state, sort_keys=True))
    else:
        print_text(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

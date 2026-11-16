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

"""Deterministic dataset loading shared by legacy and NVFlare runners."""

import copy
import json
from pathlib import Path


def load_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_task_results(rows: list[dict]) -> dict:
    completed = len(rows)
    successes = sum(bool(row["done"]) for row in rows)
    return {
        "accuracy": successes / completed if completed else 0.0,
        "avg_reward": sum(float(row["reward"]) for row in rows) / completed if completed else 0.0,
        "num_tasks": completed,
        "num_completed": completed,
        "num_success": successes,
        "per_task": rows,
    }


def _pddl_game_name(row: dict) -> str:
    value = row.get("game_name")
    if value is not None and str(value).strip():
        return str(value).strip().lower()
    additional_info = row.get("additional_info")
    if isinstance(additional_info, dict):
        value = additional_info.get("subtask")
        if value is not None and str(value).strip():
            return str(value).strip().lower()
    return ""


def normalize_pddl_rows(rows: list[dict]) -> list[dict]:
    per_game_next: dict[str, int] = {}
    normalized = []
    for source in rows:
        row = copy.deepcopy(source)
        game_name = _pddl_game_name(row)
        if not game_name:
            continue
        row["game_name"] = game_name
        try:
            problem_index = int(row["problem_index"])
        except (KeyError, TypeError, ValueError):
            problem_index = per_game_next.get(game_name, 0)
            per_game_next[game_name] = problem_index + 1
        row["problem_index"] = problem_index
        row["difficulty"] = "" if row.get("difficulty") is None else str(row["difficulty"])
        row.setdefault("env_name", "pddl")
        row.setdefault("task_type", "pddl")
        normalized.append(row)
    return normalized


def build_fever_task(row: dict) -> dict:
    claim = str(row.get("claim", "")).strip()
    label = str(row.get("label", "")).strip()
    if not claim or not label:
        return {}
    task = {"task": claim, "answer": label, "env_name": "fever", "memco_domain": "fever"}
    domain = str(row.get("ab_domain", "")).strip()
    if domain:
        task["ab_domain"] = domain
    return task


def split_fever_by_domain(rows: list[dict], domains: list[str]) -> dict[str, list[dict]]:
    grouped = {domain: [] for domain in domains}
    for row in rows:
        domain = str(row.get("ab_domain", "")).strip()
        if domain not in grouped:
            continue
        task = build_fever_task(row)
        if task:
            grouped[domain].append(task)
    return grouped


def load_subset_file(subset_dir: Path, domain: str, split_name: str) -> list[dict]:
    path = subset_dir / f"{domain}__{split_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing subset file: {path}")
    with path.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f"Subset must be a list: {path}")
    return rows

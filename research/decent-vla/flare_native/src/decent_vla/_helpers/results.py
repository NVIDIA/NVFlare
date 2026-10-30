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

"""Evaluation result types and results.json helpers.

Overall success is computed from total successes and episodes so suites
with different sizes are weighted correctly.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.2"


@dataclass
class TaskResult:
    task_id: int
    task_name: str
    n_episodes: int
    n_success: int
    success_rate: float
    mean_episode_length: float = 0.0


@dataclass
class BenchmarkSuiteResult:
    n_episodes: int
    n_success: int
    success_rate: float
    per_task: list[TaskResult] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    benchmark: str
    suites: dict[str, BenchmarkSuiteResult]
    overall_n_episodes: int
    overall_n_success: int
    overall_success_rate: float


@dataclass
class DeviceInfo:
    type: str
    name: str
    count: int = 1


@dataclass
class RunResults:
    """Top-level results JSON."""

    schema_version: str
    run_id: str
    backend: str
    checkpoint: str | None
    benchmark: str
    config_hash: str
    git_commit: str | None
    started_at: str
    finished_at: str
    wall_time_s: float
    device: DeviceInfo
    n_episodes_per_task: int
    seed: int
    suites: dict[str, dict[str, Any]]
    overall: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)
    # v1.1: seen/held-out aggregates + partition provenance (defaults keep
    # 1.0 files readable).
    aggregates: dict[str, Any] = field(default_factory=dict)
    partition_manifest_hash: str | None = None
    # v1.2: checkpoint/horizon/config identity, so the artifact answers "what
    # exactly was evaluated" without the job log (2026-08-14 audit gap).
    provenance: dict[str, Any] | None = None


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def config_hash(config: dict | str) -> str:
    """SHA256 of the canonicalized config. Accepts a dict or a JSON string."""
    if isinstance(config, dict):
        config = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(config.encode("utf-8")).hexdigest()


def git_commit(repo_dir: str | Path | None = None) -> str | None:
    """Best-effort short git commit of `repo_dir` (or CWD). Returns None if not a git repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def write_results(path: str | Path, results: RunResults) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(results)
    # Atomic write (tmp + rename), like the per-suite eval JSONs and the server
    # checkpoints — a crash mid-write never leaves a torn results.json.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=False)
    tmp.replace(path)


def tag_suites(suites: dict, heldout_eval_ids: dict[str, set[int]]) -> dict:
    """Attach tags=['seen'|'held_out'] to every per_task entry (spec §8)."""
    for sname, s in suites.items():
        held = heldout_eval_ids.get(sname, set())
        for t in s.get("per_task", []):
            t["tags"] = ["held_out"] if t["task_id"] in held else ["seen"]
    return suites


def compute_aggregates(suites: dict) -> dict[str, Any]:
    """totals/episodes, never average-of-averages (spec §8)."""
    buckets = {"overall": [], "seen": [], "held_out": []}
    for s in suites.values():
        for t in s.get("per_task", []):
            buckets["overall"].append(t)
            for tag in t.get("tags", []):
                if tag in buckets:
                    buckets[tag].append(t)

    def _agg(tasks, with_ntasks=False):
        n_eps = sum(t["n_episodes"] for t in tasks)
        n_suc = sum(t["n_success"] for t in tasks)
        out = {"n_episodes": n_eps, "n_success": n_suc, "success_rate": (n_suc / n_eps) if n_eps else None}
        if with_ntasks:
            out["n_tasks"] = len(tasks)
        return out

    return {
        "overall": _agg(buckets["overall"]),
        "seen": _agg(buckets["seen"], with_ntasks=True),
        "held_out": _agg(buckets["held_out"], with_ntasks=True),
    }


def now_iso() -> str:
    return _now_utc_iso()

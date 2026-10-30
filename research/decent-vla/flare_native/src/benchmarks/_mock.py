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

"""Run deterministic CPU-only evaluation episodes against a policy.

Synthetic, deterministic, CPU-only. Episodes succeed when ‖action‖² is
below a per-task threshold derived from the seed. Used to check the evaluation
pipeline without downloading models or starting LIBERO.

Real benchmarks (LIBERO, …) follow the same `evaluate(policy, suites,
n_episodes_per_task, seed, progress_cb)` shape.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from src.configs.schemas import BenchmarkConfig
from src.support.results import BenchmarkResult, BenchmarkSuiteResult, TaskResult


class MockBenchmark:
    name = "mock"
    # Each "suite" has 2 tasks; users can request "default" or "extra" or both.
    suites = ["default", "extra"]

    _TASKS_BY_SUITE = {
        "default": [(0, "mock-task-0"), (1, "mock-task-1")],
        "extra": [(2, "mock-task-2"), (3, "mock-task-3")],
    }

    EPISODE_LENGTH = 4
    SUCCESS_THRESHOLD = 100.0  # ‖action‖² under this counts as success

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg: BenchmarkConfig) -> MockBenchmark:
        unknown = set(cfg.suites) - set(cls.suites)
        if unknown:
            raise ValueError(f"Unknown MockBenchmark suite(s): {sorted(unknown)}. " f"Valid: {cls.suites}")
        return cls(cfg)

    def task_names(self, suite: str) -> dict[int, str]:
        """task_id -> task name, for mapping dataset task names onto ids
        (personalized eval). Real benchmarks expose the same accessor."""
        return dict(self._TASKS_BY_SUITE[suite])

    def _suite_tasks(self, suite: str) -> list[tuple[int, str]]:
        """Tasks of a suite, narrowed by cfg.extra['task_filter'] when set
        ({suite: [task_ids]} — the personalized-eval per-client subset)."""
        tasks = self._TASKS_BY_SUITE[suite]
        task_filter = (self.cfg.extra or {}).get("task_filter")
        if task_filter is None or suite not in task_filter:
            return tasks
        wanted = set(task_filter[suite])
        return [t for t in tasks if t[0] in wanted]

    def evaluate(
        self,
        policy: Any,
        suites: list[str],
        n_episodes_per_task: int,
        seed: int,
        progress_cb: Callable[[str, int, int, bool], None] | None = None,
    ) -> BenchmarkResult:
        rng = np.random.default_rng(seed)
        suite_results: dict[str, BenchmarkSuiteResult] = {}
        total_eps = 0
        total_succ = 0

        for suite in suites:
            tasks = self._suite_tasks(suite)
            per_task: list[TaskResult] = []
            suite_eps = 0
            suite_succ = 0
            for task_id, task_name in tasks:
                t_succ = 0
                t_lengths = []
                for ep_i in range(n_episodes_per_task):
                    policy.reset({"task_id": task_id, "seed": int(rng.integers(1 << 30))})
                    cum_norm_sq = 0.0
                    for _ in range(self.EPISODE_LENGTH):
                        # Canonical observation (design §6.1) — the mock pair
                        # exercises the same boundary as real benchmarks.
                        obs = {
                            "images": {"camera": np.zeros((8, 8, 3), np.uint8)},
                            "state": rng.standard_normal(16).astype(np.float32),
                            "instruction": task_name,
                        }
                        a = policy.select_action(obs)
                        cum_norm_sq += float(np.square(a).sum())
                    ok = cum_norm_sq < self.SUCCESS_THRESHOLD
                    if ok:
                        t_succ += 1
                    t_lengths.append(self.EPISODE_LENGTH)
                    if progress_cb is not None:
                        progress_cb(suite, task_id, ep_i, ok)
                per_task.append(
                    TaskResult(
                        task_id=task_id,
                        task_name=task_name,
                        n_episodes=n_episodes_per_task,
                        n_success=t_succ,
                        success_rate=t_succ / n_episodes_per_task if n_episodes_per_task else 0.0,
                        mean_episode_length=(sum(t_lengths) / len(t_lengths)) if t_lengths else 0.0,
                    )
                )
                suite_eps += n_episodes_per_task
                suite_succ += t_succ
            suite_results[suite] = BenchmarkSuiteResult(
                n_episodes=suite_eps,
                n_success=suite_succ,
                success_rate=suite_succ / suite_eps if suite_eps else 0.0,
                per_task=per_task,
            )
            total_eps += suite_eps
            total_succ += suite_succ

        return BenchmarkResult(
            benchmark=self.name,
            suites=suite_results,
            overall_n_episodes=total_eps,
            overall_n_success=total_succ,
            overall_success_rate=total_succ / total_eps if total_eps else 0.0,
        )

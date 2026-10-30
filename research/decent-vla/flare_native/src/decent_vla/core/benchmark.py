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

"""The `Benchmark` protocol and the data classes its result schema uses.

A Benchmark takes a `Policy` (not a `Backend`) so the same benchmark adapter
runs unchanged across every VLA backend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from decent_vla.configs.schemas import BenchmarkConfig
    from decent_vla.core.policy import Policy


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
    """What `Benchmark.evaluate` returns. Written through `core/results.py`."""

    benchmark: str
    suites: dict[str, BenchmarkSuiteResult]
    overall_n_episodes: int
    overall_n_success: int
    overall_success_rate: float


@runtime_checkable
class Benchmark(Protocol):
    name: str
    suites: list[str]

    @classmethod
    def from_config(cls, cfg: BenchmarkConfig) -> Benchmark:
        pass

    def evaluate(
        self,
        policy: Policy,
        suites: list[str],
        n_episodes_per_task: int,
        seed: int,
        # progress_cb args: (suite, task_id, episode, success)
        progress_cb: Callable[[str, int, int, bool], None] | None = None,
    ) -> BenchmarkResult:
        pass

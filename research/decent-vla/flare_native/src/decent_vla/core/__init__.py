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

"""Core protocols, types, and schemas. No backend-specific imports allowed here."""

from __future__ import annotations

from decent_vla.core.backend import Backend
from decent_vla.core.benchmark import Benchmark, BenchmarkResult, BenchmarkSuiteResult, TaskResult
from decent_vla.core.policy import Policy
from decent_vla.core.registry import Registry, discover, registry
from decent_vla.core.results import (
    RunResults,
    config_hash,
    git_commit,
    read_results,
    write_results,
)
from decent_vla.core.types import Batch, CheckpointSpec, LossDict, ParamScope

__all__ = [
    "Backend",
    "Benchmark",
    "BenchmarkResult",
    "BenchmarkSuiteResult",
    "TaskResult",
    "Policy",
    "Batch",
    "CheckpointSpec",
    "LossDict",
    "ParamScope",
    "Registry",
    "registry",
    "discover",
    "RunResults",
    "read_results",
    "write_results",
    "config_hash",
    "git_commit",
]

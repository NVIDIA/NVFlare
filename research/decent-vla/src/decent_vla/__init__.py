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

"""decent-vla — backend-agnostic federated learning for VLA policies."""
from __future__ import annotations

__version__ = "0.2.0"

from decent_vla.core.backend import Backend
from decent_vla.core.benchmark import Benchmark, BenchmarkResult
from decent_vla.core.policy import Policy
from decent_vla.core.strategy import (
    ClientContext,
    ClientUpdate,
    RoundHistory,
    RoundPlan,
    Strategy,
)
from decent_vla.core.types import Batch, CheckpointSpec, LossDict, ParamScope

__all__ = [
    "__version__",
    "Backend",
    "Benchmark",
    "BenchmarkResult",
    "Policy",
    "Strategy",
    "ClientContext",
    "ClientUpdate",
    "RoundHistory",
    "RoundPlan",
    "Batch",
    "CheckpointSpec",
    "LossDict",
    "ParamScope",
]

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

"""The `Policy` protocol — what `Backend.build_policy()` returns.

Benchmarks call `policy.reset(task_meta)` at episode start, then
`policy.select_action(observation)` per timestep. Backend-specific
observation/action conversion lives inside the Policy implementation,
reusing the adapter's training-side preprocessing.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    def reset(self, task_meta: dict) -> None:
        pass

    def select_action(self, observation: dict) -> np.ndarray:
        pass

    @property
    def action_horizon(self) -> int:
        pass

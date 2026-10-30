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

"""The mock pair must speak the canonical observation contract (design
§6.1), so the CPU suite exercises the benchmark↔policy boundary end-to-end."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from decent_vla.adapters._mock.policy import MockPolicy
from decent_vla.benchmarks._mock import MockBenchmark
from decent_vla.configs.schemas import BenchmarkConfig


def test_mock_benchmark_emits_canonical_observation():
    seen = []

    class _Recorder:
        action_horizon = 1

        def reset(self, task_meta):
            pass

        def select_action(self, observation):
            seen.append(observation)
            return np.zeros(7)

    b = MockBenchmark.from_config(BenchmarkConfig(name="mock", suites=["default"]))
    b.evaluate(_Recorder(), suites=["default"], n_episodes_per_task=1, seed=0)

    obs = seen[0]
    assert set(obs) == {"images", "state", "instruction"}
    assert isinstance(obs["images"], dict)
    for img in obs["images"].values():
        assert img.dtype == np.uint8 and img.ndim == 3
    assert obs["state"].ndim == 1
    assert isinstance(obs["instruction"], str) and obs["instruction"]


def test_mock_policy_consumes_canonical_observation():
    torch.manual_seed(0)
    model = nn.Linear(16, 7)
    policy = MockPolicy(model, in_dim=16, out_dim=7, device=torch.device("cpu"))
    policy.reset({"task_id": 0, "seed": 1})

    state = np.linspace(-1, 1, 16, dtype=np.float32)
    obs = {
        "images": {"camera": np.zeros((8, 8, 3), np.uint8)},
        "state": state,
        "instruction": "mock-task-0",
    }
    action = policy.select_action(obs)
    expected = model(torch.from_numpy(state).unsqueeze(0)).detach().numpy()[0]
    np.testing.assert_allclose(action, expected, atol=1e-6)

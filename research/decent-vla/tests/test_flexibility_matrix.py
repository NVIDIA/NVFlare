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

"""Any strategy x any client count x any scheme through ONE code path.

This is the flexibility contract: no isinstance branching, no per-backend
special cases in the runner. MockBackend stands in for any policy; the
lerobot adapter shares the identical path.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from decent_vla.adapters._mock.backend import MockBackend
from decent_vla.core.partitioner import partition_episodes
from decent_vla.runtime.local_runner import LocalRunner
from decent_vla.strategies.fedadam import FedAdam
from decent_vla.strategies.fedavg import FedAvg
from decent_vla.strategies.fedprox import FedProx


@pytest.mark.parametrize("strategy_cls", [FedAvg, FedProx, FedAdam])
@pytest.mark.parametrize("num_clients", [1, 2, 5])
@pytest.mark.parametrize("scheme", ["iid_episode", "task_round_robin"])
def test_matrix(strategy_cls, num_clients, scheme, backend_factory, data_cfg, backend_cfg):
    cfg = dataclasses.replace(
        data_cfg, extra={**(data_cfg.extra or {}), "n_tasks": 5, "eps_per_task": 4}
    )
    episodes = MockBackend.list_episodes(cfg)
    chunks, manifest = partition_episodes(episodes, scheme=scheme, num_clients=num_clients, seed=42)
    assert manifest.num_clients == num_clients

    runner = LocalRunner(
        backend_factory=backend_factory,
        strategy=strategy_cls(None),
        client_ids=[f"c{i}" for i in range(num_clients)],
        partition_indices=chunks,
        data_cfg=cfg,
        num_rounds=2,
        local_steps=2,
        optim_cfg=backend_cfg.optim,
        device=torch.device("cpu"),
    )
    out = runner.run()
    assert out.rounds_completed == 2
    assert all(v.dtype == torch.float32 for v in out.final_state.values())
    assert all(torch.isfinite(v).all() for v in out.final_state.values())

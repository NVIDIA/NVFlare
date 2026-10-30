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

"""Warm optimizer + global-step scheduler — the cold-Adam fix (spec §6)."""
from __future__ import annotations

import torch

from decent_vla.runtime.client_state import (
    ClientState,
    build_client_state,
    fast_forward_scheduler,
)
from decent_vla.strategies.fedavg import FedAvg


def test_build_client_state(backend_factory, data_cfg, backend_cfg):
    st = build_client_state(
        client_id="c0", backend_factory=backend_factory, data_cfg=data_cfg,
        optim_cfg=backend_cfg.optim, partition_indices=list(range(16)),
        strategy=FedAvg(None), start_round=0, local_steps=5)
    assert isinstance(st, ClientState)
    assert st.global_step == 0
    assert st.persistent == {}


def test_fast_forward_scheduler():
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0 / (1 + step))
    fast_forward_scheduler(sched, 10)
    assert sched.last_epoch == 10


def test_resume_fast_forwards(backend_factory, data_cfg, backend_cfg):
    st = build_client_state(
        client_id="c0", backend_factory=backend_factory, data_cfg=data_cfg,
        optim_cfg=backend_cfg.optim, partition_indices=list(range(16)),
        strategy=FedAvg(None), start_round=3, local_steps=10)
    assert st.global_step == 30
    if st.scheduler is not None:
        assert getattr(st.scheduler, "last_epoch", 30) == 30


def test_scheduler_advances_cumulatively_across_rounds(backend, data_cfg, backend_cfg):
    """LR continuity (spec §6): one scheduler, two run_client_round calls,
    last_epoch == 2 * local_steps. MockBackend ships no scheduler, so we
    attach a LambdaLR by hand."""
    from decent_vla.core.strategy import RoundPlan
    from decent_vla.runtime.client_loop import run_client_round

    loader = backend.build_dataloader(data_cfg, partition_indices=None)
    optimizer, _ = backend.build_optimizer(backend_cfg.optim)
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (1 + step))
    gs = backend.get_state("shared")
    strategy = FedAvg(None)
    persistent: dict = {}
    local_steps = 4
    for r in range(2):
        update = run_client_round(
            backend=backend, strategy=strategy,
            plan=RoundPlan(selected_clients=["c0"], local_steps=local_steps),
            dataloader=loader, client_id="c0", round_num=r,
            global_state=gs, optimizer=optimizer, scheduler=sched,
            persistent_state=persistent)
        assert update.extra["steps_executed"] == local_steps
    assert sched.last_epoch == 2 * local_steps

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

"""The canonical client-side training loop.

There is exactly one place in the codebase that drives the inner FL loop:
this function. Every backend uses it. Every strategy plugs into it via the
six client-side hook calls below. No isinstance branching anywhere.

"""
from __future__ import annotations

import logging
from collections.abc import Iterator

import torch

from decent_vla.core.backend import Backend
from decent_vla.core.strategy import ClientContext, ClientUpdate, RoundPlan, Strategy
from decent_vla.core.types import Batch

log = logging.getLogger(__name__)


def run_client_round(
    backend: Backend,
    strategy: Strategy,
    plan: RoundPlan,
    dataloader: Iterator[Batch],
    *,
    client_id: str,
    round_num: int,
    global_state: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: object | None = None,
    persistent_state: dict | None = None,
    model: torch.nn.Module | None = None,
    grad_clip_norm: float | None = None,
    delta_dtype: torch.dtype | None = None,
) -> ClientUpdate:
    """Run one round of local training on one client. Returns a ClientUpdate.

    The framework owns the loop; the backend owns `compute_loss(batch)`; the
    strategy owns the loss modification, gradient correction, and the final
    payload assembly.
    """
    ctx = ClientContext(
        client_id=client_id,
        round_num=round_num,
        step_in_round=0,
        backend=backend,
        model=model if model is not None else backend.model,
        optimizer=optimizer,
        scheduler=scheduler,
        global_state=global_state,
        broadcast=plan.broadcast,
        persistent_state=persistent_state if persistent_state is not None else {},
        scratch={"loss_sum": 0.0, "loss_count": 0},
        samples_seen=0,
        delta_dtype=delta_dtype,
    )

    strategy.on_round_start(ctx)

    for step in range(plan.local_steps):
        try:
            batch = next(dataloader)
        except StopIteration:
            # Backends are expected to return a cycling iterator. If it
            # exhausts, the round ends early: the actual count travels in
            # update.extra["steps_executed"], and a round that planned steps
            # but trained none raises below instead of reporting success.
            break

        ctx.step_in_round = step

        # ---- forward ----
        loss_dict = backend.compute_loss(batch)

        # ---- strategy: loss modification (FedProx / MOON / FedDyn) ----
        loss_dict = strategy.modify_loss(loss_dict, ctx)

        # ---- backward + step ----
        loss = loss_dict["loss"]
        loss.backward()
        strategy.on_after_backward(ctx)
        # Optional grad-norm clipping (lerobot's trainer clips at
        # grad_clip_norm=10 every step). Off by default; clips exactly the
        # optimized params (== trainable set), after any strategy grad correction.
        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                grad_clip_norm,
            )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None and hasattr(scheduler, "step"):
            scheduler.step()

        # ---- bookkeeping ----
        ctx.scratch["loss_sum"] += float(loss.detach())
        ctx.scratch["loss_count"] += 1
        ctx.samples_seen += _infer_batch_size(batch)

    update = strategy.on_round_end(ctx)
    _num_train_frames = getattr(backend, "num_train_frames", lambda: None)()
    if _num_train_frames is not None:
        update.num_samples = _num_train_frames
    steps_executed = int(ctx.scratch.get("loss_count", 0))
    update.extra.setdefault("steps_executed", steps_executed)
    if steps_executed == 0 and plan.local_steps > 0:
        raise ValueError(
            f"client {client_id} round {round_num} trained 0 of {plan.local_steps} planned steps: "
            "its dataloader is empty or exhausted. build_dataloader must return a non-empty, "
            "cycling iterator."
        )
    return update


def _infer_batch_size(batch: Batch) -> int:
    """Best-effort batch-size inference. Returns 1 if nothing is detectable.

    Used only for FL weighting in the default `on_round_end`. Strategies
    that need exact sample counts should compute them themselves.
    """
    for v in batch.values():
        if isinstance(v, torch.Tensor) and v.ndim >= 1:
            return int(v.shape[0])
    return 1

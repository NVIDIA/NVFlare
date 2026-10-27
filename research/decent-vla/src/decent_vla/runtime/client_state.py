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

"""Per-client state that survives FL rounds (spec §6).

Created once per client; the optimizer (with its Adam moments), the
scheduler (advancing on the CUMULATIVE global step), the cycling dataloader
and the strategy's persistent_state all live here. Rebuilding any of these
per round is the documented cold-Adam drift bug.

When ``client_ckpt_dir`` is supplied (opt-in ``fl.persist_client_state``),
``build_client_state`` restores a previously-saved client checkpoint instead
of cold-building — the resume counterpart to the server's checkpoint resume.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from decent_vla.core.backend import Backend
from decent_vla.core.strategy import ClientContext, Strategy


@dataclass
class ClientState:
    client_id: str
    backend: Backend
    optimizer: Any
    scheduler: object | None
    dataloader: Iterator
    persistent: dict[str, Any] = field(default_factory=dict)
    global_step: int = 0


def fast_forward_scheduler(scheduler: object | None, n_steps: int) -> None:
    """Advance a (stateless, step-based) torch scheduler by n_steps."""
    if scheduler is None or not hasattr(scheduler, "step"):
        return
    for _ in range(n_steps):
        scheduler.step()


def make_minimal_ctx(
    cid: str, round_num: int, global_state, persistent_state: dict[str, Any] | None = None
) -> ClientContext:
    """Tiny ClientContext-shaped object for `update_client_state` callbacks.

    Only used by strategies that need to update per-client state outside
    the training loop. The default `update_client_state` returns
    `ctx.persistent_state` verbatim, so the real per-client dict must ride
    along or it would be wiped each round.
    """
    return ClientContext(
        client_id=cid,
        round_num=round_num,
        step_in_round=-1,
        backend=None,  # type: ignore[arg-type]
        model=None,  # type: ignore[arg-type]
        optimizer=None,  # type: ignore[arg-type]
        scheduler=None,
        global_state=global_state,
        broadcast={},
        persistent_state=persistent_state if persistent_state is not None else {},
        scratch={},
    )


def _move_optimizer_state_to_param_device(optimizer) -> None:
    """After loading an optimizer state_dict (saved on CPU), move per-param
    state tensors (Adam m/v) onto each param's current device. No-op on CPU."""
    for group in optimizer.param_groups:
        for p in group["params"]:
            st = optimizer.state.get(p)
            if not st:
                continue
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(p.device)


def build_client_state(
    client_id: str,
    backend_factory,
    data_cfg,
    optim_cfg,
    partition_indices: list[int],
    strategy: Strategy,
    start_round: int = 0,
    local_steps: int = 0,
    client_ckpt_dir: str | Path | None = None,
    device=None,
) -> ClientState:
    # `device` (opt-in) pins this client's backend to a specific CUDA device for
    # one-GPU-per-client spread; None uses the factory's default device. Set the
    # current CUDA device first so construction-time current-device ops (tied
    # embeddings, stray `.cuda()`) also land on the pinned GPU.
    if device is not None and getattr(device, "type", None) == "cuda":
        import torch as _torch
        _torch.cuda.set_device(device)
    backend = backend_factory(device) if device is not None else backend_factory()
    dataloader = backend.build_dataloader(data_cfg, partition_indices)
    optimizer, scheduler = backend.build_optimizer(optim_cfg)
    start_step = start_round * local_steps
    fast_forward_scheduler(scheduler, start_step)
    persistent = strategy.init_client_state(client_id)
    global_step = start_step

    # Opt-in warm resume: restore the NEWEST client checkpoint from a round
    # BEFORE this run's start_round. Blobs at rounds >= start_round (a crash
    # landed between client-save(r) and server-save(r)) are skipped — but we
    # fall back to older eligible blobs rather than giving up: silently
    # cold-resetting wipes every personal parameter on personalization arms.
    if client_ckpt_dir is not None:
        from decent_vla.runtime.client_checkpoint import (
            client_checkpoints,
            load_client_state,
        )

        for blob_round, path in reversed(client_checkpoints(client_ckpt_dir, client_id)):
            if blob_round >= start_round:
                continue
            blob = load_client_state(path)
            backend.set_state(blob["weights"], scope="all")
            optimizer.load_state_dict(blob["optimizer"])
            _move_optimizer_state_to_param_device(optimizer)
            if scheduler is not None and blob.get("scheduler") is not None:
                scheduler.load_state_dict(blob["scheduler"])
            if blob.get("persistent") is not None:
                persistent = blob["persistent"]
            global_step = int(blob["global_step"])
            break

    return ClientState(
        client_id=client_id,
        backend=backend,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        persistent=persistent,
        global_step=global_step,
    )

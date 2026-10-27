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

"""Local-mode FL runner — runs the whole client/server cycle in one process.

No FL framework required; useful for:
  * smoke-testing strategies against MockBackend (`make smoke`)
  * centralized baselines (num_clients=1)
  * fast iteration on hook ordering and config validation
  * CI

This is a thin shell around the shared server loop (`runtime/round_loop.py`)
driven through a `LocalCommunicator`: the same `run_rounds` that the NVFlare
ServerApp uses, with clients realized as in-process callbacks. Per-client
state (backend, optimizer with its Adam moments, scheduler, cycling
dataloader, strategy persistent_state) lives in a `ClientState` that is
built ONCE and reused every round — rebuilding it per round is the
documented cold-Adam drift bug (spec §6).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from decent_vla.core.strategy import ClientUpdate, RoundPlan, Strategy
from decent_vla.runtime.client_loop import run_client_round
from decent_vla.runtime.client_state import ClientState, fast_forward_scheduler, make_minimal_ctx

log = logging.getLogger(__name__)


def _move_client_state(st, device) -> None:
    """Move a client's backend model + optimizer moment tensors to `device`.

    The optimizer holds references to the model Parameters (moved in-place by
    ``model.to``), but its per-parameter state tensors (Adam moments) are
    separate and must be moved explicitly to stay on the params' device.
    """
    model = getattr(st.backend, "model", None)
    if model is not None:
        model.to(device)
    for p_state in st.optimizer.state.values():
        for k, v in p_state.items():
            if torch.is_tensor(v):
                p_state[k] = v.to(device)


@dataclass
class LocalRunOutcome:
    final_state: dict[str, torch.Tensor]
    per_round_metrics: list[dict[str, Any]] = field(default_factory=list)
    rounds_completed: int = 0


class LocalRunner:
    """One process; num_clients ≥ 1; sequential local training per round."""

    def __init__(
        self,
        backend_factory,                     # () -> Backend (one persistent backend per client)
        strategy: Strategy,
        client_ids: list[str],
        partition_indices: list[list[int]],  # len(...) == len(client_ids)
        data_cfg,
        num_rounds: int,
        local_steps: int,
        optim_cfg,
        device: torch.device,
        log_every: int = 1,
        save_every_n_rounds: int | None = None,
        checkpoint_cb=None,
        min_clients: int | None = None,
        round_timeout_s: float | None = None,
        start_round: int = 0,
        initial_state: dict[str, torch.Tensor] | None = None,
        persist_client_state: bool = False,
        client_ckpt_dir: str | None = None,
        client_state_keep: int | None = None,
        client_devices: list[torch.device] | None = None,
        lean_host_memory: bool = False,
    ):
        if len(client_ids) != len(partition_indices):
            raise ValueError(
                f"client_ids ({len(client_ids)}) and partition_indices "
                f"({len(partition_indices)}) must align."
            )
        self.backend_factory = backend_factory
        self.strategy = strategy
        self.client_ids = client_ids
        self.partition_indices = partition_indices
        self.data_cfg = data_cfg
        self.num_rounds = num_rounds
        self.local_steps = local_steps
        self.optim_cfg = optim_cfg
        self.device = device
        self.log_every = log_every
        self.save_every_n_rounds = save_every_n_rounds
        self.checkpoint_cb = checkpoint_cb
        self.min_clients = min_clients
        self.round_timeout_s = round_timeout_s
        self.start_round = start_round
        self.initial_state = initial_state  # resume: replaces fresh-backend bootstrap
        self.lean_host_memory = lean_host_memory  # see RoundLoopConfig.lean_host_memory
        self.persist_client_state = persist_client_state
        self.client_ckpt_dir = client_ckpt_dir  # where per-client states are saved/restored
        self.client_state_keep = client_state_keep  # retain N highest-round files per client
        # Opt-in client-per-GPU spread: one dedicated CUDA device per client so
        # each client's full backend (model + warm Adam) stays resident on its
        # own GPU — no single-GPU juggling. For full-model FL where one client
        # already fills a GPU (pi0.5/smolvla full-FT). None => single-GPU
        # juggling (the default; unchanged for LIBERO 8-client-on-1-GPU runs).
        if client_devices is not None and len(client_devices) != len(client_ids):
            raise ValueError(
                f"client_devices ({len(client_devices)}) must align with "
                f"client_ids ({len(client_ids)})."
            )
        self.client_devices = client_devices
        self._dev_by_client = (
            dict(zip(client_ids, client_devices, strict=True))
            if client_devices is not None
            else {}
        )

        # Built lazily on first selection, then reused across rounds.
        self.client_states: dict[str, ClientState] = {}
        # OOM guard: which client's backend is currently resident on the GPU.
        self._gpu_cid: str | None = None

    def _place_on_gpu(self, cid: str) -> None:
        """Keep only the active client's backend resident on the GPU.

        LocalRunner builds one persistent backend per client (warm Adam +
        cycling loader) and reuses it across rounds. For a multi-GB model on a
        single GPU (e.g. GR00T-3B x 8 clients) keeping every client's backend
        resident OOMs the device, so offload idle clients to CPU and bring only
        the active one back. No-op on CPU/MPS and for single-client runs.
        """
        # Spread mode: every client is pinned to its own dedicated GPU and
        # stays resident there for the whole run — no juggling needed. Point the
        # current CUDA device at this client's GPU so any current-device-relative
        # op inside the model/preprocessor lands there instead of cuda:0.
        if self.client_devices is not None:
            dev = self._dev_by_client.get(cid)
            if dev is not None and dev.type == "cuda":
                torch.cuda.set_device(dev)
            self._gpu_cid = cid
            return
        if self.device.type != "cuda" or len(self.client_states) <= 1:
            self._gpu_cid = cid
            return
        if self._gpu_cid == cid:
            return
        prev = self.client_states.get(self._gpu_cid) if self._gpu_cid else None
        if prev is not None:
            _move_client_state(prev, torch.device("cpu"))
            torch.cuda.empty_cache()
        _move_client_state(self.client_states[cid], self.device)
        self._gpu_cid = cid

    def run(self) -> LocalRunOutcome:
        from decent_vla.core.communicator import LocalCommunicator
        from decent_vla.runtime.client_state import build_client_state
        from decent_vla.runtime.round_loop import RoundLoopConfig, run_rounds

        if self.initial_state is not None:
            initial = self.initial_state
            if self.lean_host_memory:
                # run_rounds will DRAIN this dict to avoid holding the bootstrap
                # state and the fp32 master at once. Hand over ownership so the
                # tensors are actually reclaimable — otherwise this attribute
                # keeps them alive and the saving is illusory. Makes run()
                # single-shot under lean mode (a second call re-bootstraps).
                self.initial_state = None
        else:
            bootstrap = self.backend_factory()
            initial = bootstrap.get_state("shared")
            # Free the bootstrap backend's full model off the GPU before the
            # per-client backends are built (otherwise it lingers resident).
            del bootstrap
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        idx_by_client = dict(zip(self.client_ids, self.partition_indices, strict=True))

        def client_fn(
            cid: str, global_state: dict[str, torch.Tensor], plan: RoundPlan
        ) -> ClientUpdate:
            st = self.client_states.get(cid)
            if st is None:
                st = build_client_state(
                    client_id=cid,
                    backend_factory=self.backend_factory,
                    data_cfg=self.data_cfg,
                    optim_cfg=self.optim_cfg,
                    partition_indices=idx_by_client[cid],
                    strategy=self.strategy,
                    start_round=self.start_round,
                    local_steps=self.local_steps,
                    client_ckpt_dir=self.client_ckpt_dir if self.persist_client_state else None,
                    device=self._dev_by_client.get(cid),
                )
                self.client_states[cid] = st
            # Partial participation: a client skipped by client-selection in
            # earlier rounds arrives with a stale scheduler. Fast-forward by
            # the gap to round_num * configured local_steps (self.local_steps
            # — plan.local_steps may be overridden per-round) so spec §6
            # cumulative-LR stays exact for skipped clients.
            expected_step = plan.round_num * self.local_steps
            if st.global_step < expected_step:
                fast_forward_scheduler(st.scheduler, expected_step - st.global_step)
                st.global_step = expected_step
            self._place_on_gpu(cid)
            st.backend.set_state(global_state, scope="shared")
            update = run_client_round(
                backend=st.backend,
                strategy=self.strategy,
                plan=plan,
                dataloader=st.dataloader,
                client_id=cid,
                round_num=plan.round_num,
                global_state=global_state,
                optimizer=st.optimizer,
                scheduler=st.scheduler,
                persistent_state=st.persistent,
                grad_clip_norm=self.optim_cfg.grad_clip_norm,
            )
            # Advance by the ACTUAL steps executed (an exhausted dataloader
            # ends the round early), not the planned count.
            st.global_step += update.extra.get("steps_executed", plan.local_steps)
            update.client_id = cid
            ctx = make_minimal_ctx(cid, plan.round_num, global_state, st.persistent)
            st.persistent = self.strategy.update_client_state(cid, ctx, update)
            # Opt-in: persist this client's full state at the server's cadence
            # (and on the final round) so a killed run resumes warm.
            if self.persist_client_state and self.client_ckpt_dir is not None:
                n = self.save_every_n_rounds
                is_final = (plan.round_num + 1) >= (self.start_round + self.num_rounds)
                if (n and (plan.round_num + 1) % n == 0) or is_final:
                    from decent_vla.runtime.client_checkpoint import save_client_state

                    save_client_state(
                        self.client_ckpt_dir, st, round_idx=plan.round_num,
                        keep=self.client_state_keep,
                    )
            return update

        outcome = run_rounds(
            strategy=self.strategy,
            communicator=LocalCommunicator(client_fn=client_fn),
            initial_state=initial,
            client_ids=self.client_ids,
            loop_cfg=RoundLoopConfig(
                num_rounds=self.num_rounds,
                local_steps=self.local_steps,
                min_clients=self.min_clients,
                round_timeout_s=self.round_timeout_s,
                save_every_n_rounds=self.save_every_n_rounds,
                start_round=self.start_round,
                lean_host_memory=self.lean_host_memory,
            ),
            checkpoint_cb=self.checkpoint_cb,
            log_every=self.log_every,
        )
        return LocalRunOutcome(
            final_state=outcome.final_state,
            per_round_metrics=outcome.per_round_metrics,
            rounds_completed=outcome.rounds_completed,
        )

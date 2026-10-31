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

"""The `Strategy` protocol — eight hooks covering aggregation, local-loop,
and personalized FL.

Most strategies override 1-3 hooks. FedAvg overrides 0 (the defaults *are*
FedAvg). Personalized strategies (Ditto-class) also override the two
`*_client_state` hooks.

Strategy code never imports an FL framework. The NVFlare runtime in
`runtime/nvflare/` carries our `ClientUpdate` and `Strategy` over NVFlare.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

from decent_vla.core.types import LossDict

if TYPE_CHECKING:
    from decent_vla.core.backend import Backend


# --- Server-side types --------------------------------------------------------


@dataclass
class RoundHistory:
    """Read-only view of past rounds, passed into `configure_round`."""

    round_num: int
    client_losses: dict[str, list[float]] = field(default_factory=dict)
    # Other history (per-client deltas, control variates, …) can be added
    # by individual strategies via their own per-round state.


@dataclass
class RoundPlan:
    """What `configure_round` returns."""

    selected_clients: list[str]
    local_steps: int
    broadcast: dict[str, Any] = field(default_factory=dict)
    # The framework will overlay `global_state` onto broadcast before sending
    # unless the strategy explicitly disables that.
    include_global_state: bool = True
    # Stamped by the server round loop after configure_round; clients read it
    # so the round number travels with the plan over any Communicator.
    round_num: int = 0


@dataclass
class ClientUpdate:
    """What `on_round_end` returns and `aggregate` consumes."""

    delta: dict[str, torch.Tensor]
    num_samples: int
    client_id: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)
    # `extra` carries strategy-specific payload: SCAFFOLD Δc, MOON
    # representations, etc.


# --- Client-side context ------------------------------------------------------


@dataclass
class ClientContext:
    """Per-round client state, passed into every client-side hook."""

    client_id: str
    round_num: int
    step_in_round: int
    backend: Backend
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: object | None
    global_state: dict[str, torch.Tensor]
    broadcast: dict[str, Any]
    persistent_state: dict[str, Any]
    scratch: dict[str, Any] = field(default_factory=dict)
    samples_seen: int = 0
    # Cast the round delta to this dtype ON DEVICE, before the D2H copy.
    # None = keep the master's precision (historical behaviour).
    #
    # The delta is going to be cast to the wire dtype anyway; doing it on the
    # GPU means the HOST only ever holds the small version. At pi0.5's 3.62B
    # params that is 6.7 GiB instead of 13.5 GiB per client, and it lands
    # exactly at the round boundary where every client is materialising its
    # delta at once — the measured peak host-RAM moment. Numerically
    # identical: same cast, same values, just applied one hop earlier.
    delta_dtype: torch.dtype | None = None


# --- The protocol -------------------------------------------------------------


@runtime_checkable
class Strategy(Protocol):
    name: str

    # ===== server side =====
    def configure_round(
        self, round_num: int, client_ids: list[str], history: RoundHistory
    ) -> RoundPlan: ...

    def aggregate(
        self,
        client_updates: list[ClientUpdate],
        global_state: dict[str, torch.Tensor],
        consume: bool = False,
        inplace: bool = False,
    ) -> dict[str, torch.Tensor]: ...

    # ===== client side =====
    def on_round_start(self, ctx: ClientContext) -> None: ...
    def modify_loss(self, loss: LossDict, ctx: ClientContext) -> LossDict: ...
    def on_after_backward(self, ctx: ClientContext) -> None: ...
    def on_round_end(self, ctx: ClientContext) -> ClientUpdate: ...

    # ===== personalized (Ditto, FedRep, pFedMe) =====
    def init_client_state(self, client_id: str) -> dict[str, Any]: ...
    def update_client_state(
        self, client_id: str, ctx: ClientContext, update: ClientUpdate
    ) -> dict[str, Any]: ...


# --- Base class providing default behaviours (FedAvg semantics) ---------------


class StrategyBase:
    """Default implementations of every Strategy hook.

    Concrete strategies subclass this and override only the hooks they need.
    Using a base class (rather than the Protocol itself) for implementations
    means hook defaults live in one place — strategies stay terse.
    """

    name: str = "base"

    # ----- server side -----

    def configure_round(
        self, round_num: int, client_ids: list[str], history: RoundHistory
    ) -> RoundPlan:
        """local_steps=0 is a sentinel meaning 'use the runner's configured
        local_steps'; return a positive value only to override it."""
        return RoundPlan(selected_clients=list(client_ids), local_steps=0)

    def aggregate(
        self,
        client_updates: list[ClientUpdate],
        global_state: dict[str, torch.Tensor],
        consume: bool = False,
        inplace: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Weighted-by-sample-count averaging of client deltas, applied to global_state.

        Pure by default (`consume=False`): caller-owned `client_updates` are left
        untouched. Pass `consume=True` only when you own the fresh, per-round
        updates and are about to discard them anyway — this frees each client's
        delta tensors as soon as they're folded in, which matters for large models.

        `inplace=True` additionally licenses mutating `global_state` itself:
        deltas accumulate straight into the caller's tensors and the SAME dict is
        returned, so the fp32 master is never duplicated. Numerically identical to
        the copying path (same `add_`, same order, same dtypes), but it destroys
        the caller's previous global state — pass it only when you rebind to the
        return value and hold no other reference. At pi0.5's 3.62B params the
        clone it removes is 13.5 GB, which on a 125 GB host is the difference
        between fitting and not. Driven by `fl.lean_host_memory`; the method
        default stays False so the pure contract is unchanged for direct callers.
        """
        if not client_updates:
            return global_state

        total_samples = sum(u.num_samples for u in client_updates)
        if total_samples == 0:
            return global_state

        # `global_state` is always already detached on the server paths
        # (round_loop upcasts with .detach()), so in-place add_ is safe.
        new_state = (
            global_state if inplace
            else {k: v.detach().clone() for k, v in global_state.items()}
        )
        for update in client_updates:
            w = update.num_samples / total_samples
            for k, delta in update.delta.items():
                if k in new_state:
                    new_state[k].add_(delta.to(new_state[k].dtype).to(new_state[k].device), alpha=w)
            if consume:
                update.delta.clear()  # free this client's tensors as soon as they're folded in
        return new_state

    # ----- client side -----

    def on_round_start(self, ctx: ClientContext) -> None:
        return None

    def modify_loss(self, loss: LossDict, ctx: ClientContext) -> LossDict:
        return loss

    def on_after_backward(self, ctx: ClientContext) -> None:
        return None

    def on_round_end(self, ctx: ClientContext) -> ClientUpdate:
        """Default: delta = current_shared_state - global_state at round start.

        The subtraction happens on the client's device; `ctx.delta_dtype` (when
        set) narrows each tensor there, so the host never materialises a
        full-precision copy of the delta.
        """
        current = ctx.backend.get_state("shared")
        delta = {}
        for k, v in current.items():
            if k in ctx.global_state:
                d = v.detach() - ctx.global_state[k].to(v.device)
                if ctx.delta_dtype is not None and d.is_floating_point():
                    d = d.to(ctx.delta_dtype)
                delta[k] = d.cpu()
                del d
        metrics = {}
        if "loss_sum" in ctx.scratch and ctx.scratch.get("loss_count", 0) > 0:
            metrics["loss"] = ctx.scratch["loss_sum"] / ctx.scratch["loss_count"]
        return ClientUpdate(
            delta=delta,
            num_samples=ctx.samples_seen,
            metrics=metrics,
        )

    # ----- resumable server-side state -----

    def state_dict(self) -> dict[str, Any]:
        """Server-side aggregator state to persist across a resume.

        Stateless strategies (FedAvg, FedProx) have none — the default is empty.
        FedAdam overrides this to carry its moment buffers + step counter so a
        resumed run doesn't zero them (which would take a
        bias-corrected, momentum-wiped first step). Persisted alongside the
        server checkpoint blob and restored by the runtime before round 0.
        """
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None

    # ----- personalized -----

    def init_client_state(self, client_id: str) -> dict[str, Any]:
        return {}

    def update_client_state(
        self, client_id: str, ctx: ClientContext, update: ClientUpdate
    ) -> dict[str, Any]:
        return ctx.persistent_state

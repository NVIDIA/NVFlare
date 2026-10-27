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

"""The Communicator protocol — the seam between algorithm and execution planes.

Server-side methods (broadcast_state/collect_updates) are used by
runtime/round_loop.py; client-side methods (receive_round/send_update) by
remote client processes. LocalCommunicator collapses both sides into
synchronous in-process calls. v0.2 topology is always star; ring/mesh are
v0.4 (spec §13).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import torch

from decent_vla.core.strategy import ClientUpdate, RoundPlan


@dataclass(frozen=True, eq=False)
class Topology:
    kind: str = "star"  # "star" | "ring" | "mesh" (v0.4)
    neighbors: dict[str, list[str]] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Topology) and (self.kind, self.neighbors) == (
            other.kind,
            other.neighbors,
        )

    def __hash__(self) -> int:
        return hash(self.kind)


@runtime_checkable
class Communicator(Protocol):
    # ----- server side -----
    def broadcast_state(
        self, state: dict[str, torch.Tensor], plan: RoundPlan
    ) -> None: ...
    def collect_updates(
        self, expected: list[str], timeout_s: float | None
    ) -> list[ClientUpdate]: ...
    # ----- client side -----
    def receive_round(self) -> tuple[dict[str, torch.Tensor], RoundPlan]: ...
    def send_update(self, update: ClientUpdate) -> None: ...
    # ----- common -----
    def topology(self) -> Topology: ...
    def close(self) -> None: ...


ClientFn = Callable[[str, dict[str, torch.Tensor], RoundPlan], ClientUpdate]


class LocalCommunicator:
    """In-process communicator: clients are a callback run sequentially."""

    def __init__(self, client_fn: ClientFn) -> None:
        self._client_fn = client_fn
        self._state: dict[str, torch.Tensor] = {}
        self._plan: RoundPlan | None = None

    def broadcast_state(self, state: dict[str, torch.Tensor], plan: RoundPlan) -> None:
        self._state, self._plan = state, plan

    def collect_updates(
        self, expected: list[str], timeout_s: float | None = None
    ) -> list[ClientUpdate]:
        assert self._plan is not None, "broadcast_state must precede collect_updates"
        updates = []
        for cid in expected:
            update = self._client_fn(cid, self._state, self._plan)
            if not update.client_id:
                update.client_id = cid
            updates.append(update)
        return updates

    def receive_round(self) -> tuple[dict[str, torch.Tensor], RoundPlan]:  # pragma: no cover
        raise RuntimeError("LocalCommunicator has no remote client side")

    def send_update(self, update: ClientUpdate) -> None:  # pragma: no cover
        raise RuntimeError("LocalCommunicator has no remote client side")

    def topology(self) -> Topology:
        return Topology(kind="star")

    def close(self) -> None:
        pass

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

"""Core FedRevive update policy independent of the Collab scheduler."""

import math
from dataclasses import dataclass
from enum import Enum

import torch

from nvflare.app_opt.pt.distillation.teachers import TeacherBuffer as SnapshotBuffer

ModelState = dict[str, torch.Tensor]


class Method(str, Enum):
    FEDAVG = "fedavg"
    FEDBUFF = "fedbuff"
    FEDREVIVE = "fedrevive"


class ClassProportionSource(str, Enum):
    """Select how the server obtains each client's class proportions."""

    TRUE_HISTOGRAM = "true-histogram"
    ESTIMATED = "estimated"


@dataclass(frozen=True)
class MethodConfig:
    """Paper configuration that controls scheduling and aggregation.

    concurrency bounds active logical clients; buffer_size counts accepted
    contributions per global version; min_open_slots controls refill. A refill
    threshold of one gives immediate asynchronous replacement. Setting all
    three counts equal gives the synchronous FedAvg boundary.
    """

    concurrency: int
    buffer_size: int
    min_open_slots: int
    server_lr: float


FIGURE_2_METHOD_CONFIGS = {
    Method.FEDAVG: MethodConfig(concurrency=100, buffer_size=100, min_open_slots=100, server_lr=1.4),
    Method.FEDBUFF: MethodConfig(concurrency=100, buffer_size=2, min_open_slots=1, server_lr=0.05),
    Method.FEDREVIVE: MethodConfig(concurrency=100, buffer_size=1, min_open_slots=1, server_lr=0.10),
}


@dataclass(frozen=True)
class TeacherState:
    """A returned client model and its server-derived class proxy."""

    model: ModelState
    class_proxy: torch.Tensor
    client_name: str
    base_version: int


class TeacherBuffer:
    """Shared immutable model history with research-owned per-client proxies.

    Proxies are client state: a second upload updates every retained teacher
    from that client without mutating the shared model snapshots.
    """

    def __init__(self, max_size: int):
        self.snapshots = SnapshotBuffer(max_size)
        self._proxies = {}

    @staticmethod
    def _normalize(proxy):
        proxy = proxy.detach().cpu().float().clone()
        if proxy.ndim != 1 or not torch.isfinite(proxy).all() or (proxy < 0).any() or proxy.sum() <= 0:
            raise ValueError("class_proxy must be a finite nonnegative vector with positive mass")
        return proxy / proxy.sum()

    def add(self, model, class_proxy, client_name, base_version):
        self._proxies[client_name] = self._normalize(class_proxy)
        self.snapshots.add(model, {"client_name": client_name, "base_version": int(base_version)})
        names = {item.metadata["client_name"] for item in self.snapshots}
        self._proxies = {name: self._proxies[name] for name in names}

    def __len__(self):
        return len(self.snapshots)

    def __iter__(self):
        for item in self.snapshots:
            name = item.metadata["client_name"]
            yield TeacherState(item.model_state, self._proxies[name], name, item.metadata["base_version"])

    def update_class_proxy(self, client_name, class_proxy):
        if client_name in self._proxies:
            self._proxies[client_name] = self._normalize(class_proxy)


def cosine_staleness_beta(staleness: int, max_staleness: float) -> float:
    """Return the clipped cosine FedRevive mixing weight."""

    if staleness < 0:
        raise ValueError("staleness must be >= 0")
    if max_staleness <= 0:
        raise ValueError("max_staleness must be > 0")
    if staleness > 2 * max_staleness:
        return 1.0
    beta = 1.0 - 0.5 * (1.0 + math.cos(math.pi * staleness / (2.0 * max_staleness)))
    return min(1.0, max(0.0, beta))


def model_delta(updated_model: ModelState, base_model: ModelState) -> ModelState:
    """Compute a returned model's delta relative to its assignment snapshot."""

    if updated_model.keys() != base_model.keys():
        raise ValueError("updated and base model states must have identical keys")
    return {name: updated_model[name].detach().cpu() - base_model[name].detach().cpu() for name in base_model}


def blend_updates(client_update: ModelState, distilled_update: ModelState, beta: float) -> ModelState:
    """Blend the stale client and DFKD updates before FedBuff aggregation."""

    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be between 0 and 1")
    if client_update.keys() != distilled_update.keys():
        raise ValueError("client and distilled updates must have identical keys")
    if beta == 0.0:
        return {name: value.detach().clone() for name, value in client_update.items()}
    if beta == 1.0:
        return {name: value.detach().clone() for name, value in distilled_update.items()}
    return {name: (1.0 - beta) * client_update[name] + beta * distilled_update[name] for name in client_update}

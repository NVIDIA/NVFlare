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
from collections import deque
from dataclasses import dataclass
from enum import Enum

import torch
from model import reset_tracker_stats_in_state

from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper

ModelState = dict[str, torch.Tensor]


class Method(str, Enum):
    FEDAVG = "fedavg"
    FEDBUFF = "fedbuff"
    FEDREVIVE = "fedrevive"


class FedReviveMode(str, Enum):
    """Select compatibility with the local simulator or the published method."""

    REPRODUCTION = "reproduction"
    PAPER_ALIGNED = "paper-aligned"


@dataclass(frozen=True)
class MethodConfig:
    """Paper configuration that controls scheduling and aggregation.

    K (num_active_jobs) bounds logical concurrency, B (buffer_size) controls
    how many arrivals create a global version, and O (min_open_slots) controls
    when that version is redistributed.  In particular, O=1 is asynchronous
    FedBuff while O=B=K gives the synchronous FedAvg boundary.
    """

    num_active_jobs: int
    buffer_size: int
    min_open_slots: int
    server_lr: float


FIGURE_2_METHOD_CONFIGS = {
    Method.FEDAVG: MethodConfig(num_active_jobs=100, buffer_size=100, min_open_slots=100, server_lr=1.4),
    Method.FEDBUFF: MethodConfig(num_active_jobs=100, buffer_size=2, min_open_slots=1, server_lr=0.05),
    Method.FEDREVIVE: MethodConfig(num_active_jobs=100, buffer_size=1, min_open_slots=1, server_lr=0.10),
}


@dataclass(frozen=True)
class TeacherState:
    """A returned client model and its server-derived class proxy."""

    model: ModelState
    class_proxy: torch.Tensor
    client_name: str
    base_version: int


class TeacherBuffer:
    """Keep the bounded algorithmic teacher state in most-recent-first order.

    Unlike out-of-order RPC results, these models are intentionally resident:
    FedRevive needs the latest teachers for distillation.  ``deque(maxlen=...)``
    makes that cost independent of run length and releases the oldest teacher
    as soon as a new one is inserted.
    """

    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError("teacher buffer size must be >= 1")
        self._teachers = deque(maxlen=max_size)

    def add(
        self,
        model: ModelState,
        class_proxy: torch.Tensor,
        client_name: str,
        base_version: int,
    ) -> None:
        if class_proxy.ndim != 1:
            raise ValueError("class_proxy must be a one-dimensional tensor")
        total = float(class_proxy.sum())
        if total <= 0:
            raise ValueError("class_proxy must have positive mass")
        teacher = TeacherState(
            model={name: value.detach().cpu().clone() for name, value in model.items()},
            class_proxy=class_proxy.detach().cpu().float().clone() / total,
            client_name=client_name,
            base_version=int(base_version),
        )
        self._teachers.appendleft(teacher)

    def __len__(self) -> int:
        return len(self._teachers)

    def __iter__(self):
        return iter(self._teachers)

    def update_class_proxy(self, client_name: str, class_proxy: torch.Tensor) -> None:
        """Apply a client's newest running proxy to its buffered teachers.

        A paper-aligned proxy changes once, when the client's second upload is
        averaged with its first.  Teacher entries represent model snapshots,
        but their class weights are client-level state and should immediately
        see that finalized estimate even if an older model remains buffered.
        """

        total = float(class_proxy.sum())
        if class_proxy.ndim != 1 or total <= 0:
            raise ValueError("class_proxy must be one-dimensional with positive mass")
        normalized = class_proxy.detach().cpu().float().clone() / total
        updated = deque(maxlen=self._teachers.maxlen)
        for teacher in self._teachers:
            if teacher.client_name == client_name:
                teacher = TeacherState(
                    model=teacher.model,
                    class_proxy=normalized.clone(),
                    client_name=teacher.client_name,
                    base_version=teacher.base_version,
                )
            updated.append(teacher)
        self._teachers = updated


@dataclass(frozen=True)
class BufferedUpdate:
    """One processed update waiting for the configured aggregation boundary."""

    delta: ModelState
    client_name: str
    base_version: int
    staleness: int
    beta: float


class InTimeUpdateBuffer:
    """Accumulate accepted deltas without retaining B full model dictionaries.

    This changes storage, not the aggregation boundary: the helper maintains a
    running weighted sum/count, and ``aggregate`` is still called only on the
    B-th accepted arrival.  It is especially important for synchronous FedAvg,
    where B=K=100 in the paper; retaining 100 ResNet deltas can otherwise cause
    avoidable memory pressure even though only their average is required.
    """

    def __init__(self):
        self._helper = WeightedAggregationHelper(weigh_by_local_iter=False)
        self._base_versions: list[int] = []

    def __len__(self) -> int:
        return self._helper.get_len()

    @property
    def base_versions(self) -> tuple[int, ...]:
        return tuple(self._base_versions)

    def add(self, delta: ModelState, client_name: str, base_version: int) -> None:
        self._helper.add(
            data=delta,
            weight=1.0,
            contributor_name=client_name,
            contribution_round=base_version,
        )
        self._base_versions.append(base_version)

    def aggregate(self) -> ModelState:
        result = self._helper.get_result()
        self._base_versions.clear()
        return result


@dataclass(frozen=True)
class AggregationResult:
    """A newly created global model."""

    global_model: ModelState


def cosine_staleness_beta(staleness: int, max_staleness: float) -> float:
    """Return the clipped cosine FedRevive mixing weight from the reference code."""

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


def process_update(
    method: Method | str,
    global_model: ModelState,
    updated_model: ModelState,
    base_model: ModelState,
    client_name: str,
    base_version: int,
    current_version: int,
    update_buffer: list[BufferedUpdate] | InTimeUpdateBuffer,
    distilled_update: ModelState | None = None,
    max_staleness: float = 75,
    buffer_size: int | None = None,
    server_lr: float | None = None,
) -> AggregationResult | None:
    """Accept one arrival and aggregate only at the method's Figure 2 boundary.

    FedAvg, FedBuff, and FedRevive intentionally use this same path. FedRevive
    differs only by optionally blending a KD-Revive delta into the arriving
    client delta. The buffer may retain individual deltas or accumulate them in
    time. A ``None`` return means that the configured boundary is not yet met;
    otherwise the returned result owns the new global model.
    """

    method = Method(method)
    config = FIGURE_2_METHOD_CONFIGS[method]
    buffer_size = config.buffer_size if buffer_size is None else int(buffer_size)
    server_lr = config.server_lr if server_lr is None else float(server_lr)
    if buffer_size < 1:
        raise ValueError("buffer_size must be >= 1")
    if server_lr <= 0:
        raise ValueError("server_lr must be > 0")
    if current_version < base_version:
        raise ValueError("current_version must be >= base_version")
    if len(update_buffer) >= buffer_size:
        raise RuntimeError("update buffer must be aggregated before accepting another update")

    staleness = current_version - base_version
    client_delta = model_delta(updated_model, base_model)
    beta = 0.0
    processed_delta = client_delta
    if method is Method.FEDREVIVE and distilled_update is not None:
        beta = cosine_staleness_beta(staleness, max_staleness)
        processed_delta = blend_updates(client_delta, distilled_update, beta)
    elif method is not Method.FEDREVIVE and distilled_update is not None:
        raise ValueError("distilled_update is only valid for FedRevive")

    if isinstance(update_buffer, InTimeUpdateBuffer):
        # Fold the delta into a running sum immediately, but do not publish a
        # model early.  len(update_buffer) remains the number of arrivals so the
        # B boundary is identical to retaining each BufferedUpdate.
        update_buffer.add(processed_delta, client_name, base_version)
    else:
        update_buffer.append(
            BufferedUpdate(
                delta={name: value.detach().cpu().clone() for name, value in processed_delta.items()},
                client_name=client_name,
                base_version=base_version,
                staleness=staleness,
                beta=beta,
            )
        )
    if len(update_buffer) < buffer_size:
        # A FedBuff buffer is defined by arrival order.  The scheduler is
        # responsible for making that order follow the configured delays rather
        # than nondeterministic host/RPC completion order.
        return None

    base_versions = (
        update_buffer.base_versions
        if isinstance(update_buffer, InTimeUpdateBuffer)
        else tuple(update.base_version for update in update_buffer)
    )
    if method is Method.FEDAVG and len(set(base_versions)) != 1:
        raise RuntimeError("FedAvg buffer contains updates from different global versions")

    if isinstance(update_buffer, InTimeUpdateBuffer):
        accepted_updates = ()
        average_delta = update_buffer.aggregate()
    else:
        accepted_updates = tuple(update_buffer)
        average_delta = {
            name: sum((update.delta[name] for update in accepted_updates), torch.zeros_like(global_model[name]))
            / len(accepted_updates)
            for name in global_model
        }
        update_buffer.clear()
    new_global_model = {
        name: value.detach().cpu().clone() + server_lr * average_delta[name] for name, value in global_model.items()
    }
    new_global_model = reset_tracker_stats_in_state(new_global_model)
    return AggregationResult(global_model=new_global_model)

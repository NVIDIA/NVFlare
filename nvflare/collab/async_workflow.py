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

"""Framework-independent buffered asynchronous training.

The engine is single-owner: invoke dispatch/accept on one workflow thread.
Adapters own model arithmetic; no runtime or ML framework is imported here.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper


class AsyncModelAdapter(Protocol):
    """Model states are dictionaries. Methods must not mutate supplied states."""

    def snapshot(self, model: dict) -> dict:
        """Return an independent model state suitable for storage and transfer."""

    def validate(self, model: dict, reference: dict) -> None:
        """Check keys, shapes, dtypes, and finite values against reference."""

    def difference(self, model: dict, base: dict) -> dict:
        """Return an owned delta for entries participating in aggregation."""

    def apply(self, model: dict, delta: dict, scale: float) -> dict:
        """Return the updated model, preserving other entries per adapter policy."""


@dataclass(frozen=True)
class Assignment:
    """Server-owned identity and exact starting snapshot; base_model is read-only."""

    assignment_id: int
    client_name: str
    logical_name: str
    base_version: int
    base_model: dict = field(repr=False)


@dataclass(frozen=True)
class AsyncResult:
    """One terminal client result, with an explicit full-model/delta convention.

    Delta payloads are relative to base_version. Metrics determine weight only
    when the engine explicitly configures a weight_metric.
    """

    assignment_id: int
    base_version: int
    model: dict
    metrics: dict = field(default_factory=dict)
    is_delta: bool = False

    def __post_init__(self):
        for name, value in (("assignment_id", self.assignment_id), ("base_version", self.base_version)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.is_delta, bool):
            raise ValueError("is_delta must be a boolean")
        if not isinstance(self.model, dict) or not isinstance(self.metrics, dict):
            raise TypeError("model and metrics must be dictionaries")


@dataclass(frozen=True)
class AcceptedUpdate:
    """Input to an optional callable update transform; borrowed states are read-only.

    Return a replacement delta dictionary. The client delta references the
    assignment base; current_model is the model to which aggregation is applied.
    A transform can use another reference (e.g. KD against current_model), but
    must document its semantics. Transform exceptions stop the workflow.
    """

    assignment: Assignment
    current_version: int
    current_model: dict
    delta: dict
    weight: float
    metrics: dict

    @property
    def staleness(self):
        return self.current_version - self.assignment.base_version


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


class AsyncEngine:
    """K active assignments, B accepted updates/version, O slots before refill.

    Stops after num_updates global publications, issuing no speculative work
    beyond that accepted-contribution budget. Errors are fail-fast; there is no
    automatic retry. Duplicate/stale terminal events are ignored by assignment
    identity. close() discards partial accumulation and releases base snapshots.
    """

    def __init__(
        self,
        initial_model: dict,
        model_adapter: AsyncModelAdapter,
        num_updates: int,
        num_active_jobs: int,
        buffer_size: int = 1,
        min_open_slots: int = 1,
        server_lr: float = 1.0,
        weight_metric: str | None = None,
        update_transform=None,
    ):
        for name, value in (
            ("num_updates", num_updates),
            ("num_active_jobs", num_active_jobs),
            ("buffer_size", buffer_size),
            ("min_open_slots", min_open_slots),
        ):
            _positive_int(name, value)
        if min_open_slots > num_active_jobs:
            raise ValueError("min_open_slots cannot exceed num_active_jobs")
        if not math.isfinite(server_lr) or server_lr <= 0:
            raise ValueError("server_lr must be finite and positive")
        if weight_metric is not None and (not isinstance(weight_metric, str) or not weight_metric):
            raise ValueError("weight_metric must be a nonempty string or None")
        if update_transform is not None and not callable(update_transform):
            raise TypeError("update_transform must be callable")
        self.adapter = model_adapter
        self.model = model_adapter.snapshot(initial_model)
        model_adapter.validate(self.model, self.model)
        self.num_updates = num_updates
        self.num_active_jobs = num_active_jobs
        self.buffer_size = buffer_size
        self.min_open_slots = min_open_slots
        self.server_lr = server_lr
        self.weight_metric = weight_metric
        self.update_transform = update_transform
        self.version = 0
        self.accepted_count = 0
        self.active: dict[str, Assignment] = {}
        self._next_id = 0
        self._buffer = WeightedAggregationHelper()
        self._closed = False

    @property
    def finished(self):
        return self.version >= self.num_updates

    @property
    def buffered_count(self):
        return self._buffer.get_len()

    def dispatch(self, available_clients: list[str]) -> list[Assignment]:
        """Reserve idle clients in supplied order, sharing one immutable snapshot.

        Logical names default to physical names. A caller may supply a selection
        policy by ordering/subsetting available_clients before dispatch.
        """
        if self._closed or self.finished:
            return []
        if len(set(available_clients)) != len(available_clients):
            raise ValueError("available client names must be unique")
        slots = self.num_active_jobs - len(self.active)
        if slots < self.min_open_slots:
            return []
        remaining = self.num_updates * self.buffer_size - self.accepted_count - len(self.active)
        clients = [name for name in available_clients if name not in self.active][: min(slots, remaining)]
        if not clients:
            return []
        snapshot = self.adapter.snapshot(self.model)
        assignments = []
        for name in clients:
            assignment = Assignment(self._next_id, name, name, self.version, snapshot)
            self._next_id += 1
            self.active[name] = assignment
            assignments.append(assignment)
        return assignments

    def accept(self, assignment: Assignment, result: AsyncResult | None, error: Any = None) -> bool:
        """Accept exactly once; return True only when a global version is published.

        Validate before changing state. Rejected input raises and ends the run;
        it never contributes to B. Equal weights are default; configured metric
        weights must be finite and strictly positive.
        """
        current = self.active.get(assignment.client_name)
        if self._closed or current is None or current.assignment_id != assignment.assignment_id:
            return False
        assignment = current
        if error is not None:
            raise RuntimeError(f"assignment {assignment.assignment_id} failed: {error}")
        if result is None:
            raise RuntimeError(f"assignment {assignment.assignment_id} returned no result")
        if not isinstance(result, AsyncResult):
            raise TypeError("result must be AsyncResult")
        if result.assignment_id != assignment.assignment_id or result.base_version != assignment.base_version:
            raise ValueError("assignment metadata mismatch")
        reference = (
            self.adapter.difference(assignment.base_model, assignment.base_model)
            if result.is_delta
            else assignment.base_model
        )
        self.adapter.validate(result.model, reference)
        delta = result.model if result.is_delta else self.adapter.difference(result.model, assignment.base_model)
        weight = 1.0 if self.weight_metric is None else float(result.metrics[self.weight_metric])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("aggregation weight must be finite and positive")
        if self.update_transform is not None:
            delta = self.update_transform(
                AcceptedUpdate(assignment, self.version, self.model, delta, weight, result.metrics)
            )
        self.adapter.validate(delta, self.adapter.difference(assignment.base_model, assignment.base_model))
        self._buffer.add(delta, weight, str(assignment.assignment_id), assignment.base_version)
        del self.active[assignment.client_name]
        self.accepted_count += 1
        if self.buffered_count < self.buffer_size:
            return False
        self.model = self.adapter.apply(self.model, self._buffer.get_result(), self.server_lr)
        self.version += 1
        return True

    def close(self):
        """Release outstanding bases and partial accumulation; do not publish a partial version."""
        self._closed = True
        self.active.clear()
        self._buffer.reset_stats()

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

"""Framework-independent async assignment and aggregation engine.

The engine is single-owner: invoke dispatch/accept on one workflow thread.
Adapters own model arithmetic; no runtime or ML framework is imported here.
"""

import math
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Protocol

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
    """Server-local identity and an immutable model reference until acceptance.

    The adapter returns a new state for each publication, so assignments from
    the same version safely share a base without copying or reference counts.
    """

    assignment_id: int
    client_name: str
    logical_name: str
    base_version: int
    base_model: dict = field(repr=False)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class AsyncEvent:
    """Lightweight application trace; no model states are retained."""

    phase: str
    assignment_id: int
    logical_name: str
    client_name: str
    base_version: int
    current_version: int
    metadata: dict = field(default_factory=dict)


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
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name, value in (("assignment_id", self.assignment_id), ("base_version", self.base_version)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.is_delta, bool):
            raise ValueError("is_delta must be a boolean")
        if not all(isinstance(value, dict) for value in (self.model, self.metrics, self.metadata)):
            raise TypeError("model, metrics, and metadata must be dictionaries")


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
    _returned_model: dict | Callable[[], dict] = field(repr=False)
    metadata: dict

    @cached_property
    def returned_model(self) -> dict:
        """Borrow the full upload, reconstructing a delta payload only on demand.

        Delta-only consumers need no full model. An explicit reconstruction
        uses the adapter's original dtype/buffer rules and can fail if the
        unscaled full model is unrepresentable, even when a scaled update is
        valid. This does not affect acceptance unless a consumer requests it.
        """
        return self._returned_model() if callable(self._returned_model) else self._returned_model

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
        update_observer=None,
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
        self.update_observer = update_observer
        self.version = 0
        self.accepted_count = 0
        self.active: dict[int, Assignment] = {}
        self._next_id = 0
        self._buffer = WeightedAggregationHelper()
        self._closed = False

    @property
    def finished(self):
        return self.version >= self.num_updates

    @property
    def buffered_count(self):
        return self._buffer.get_len()

    @property
    def available_slots(self):
        """Capacity allowed by K/O and the exact remaining contribution budget."""
        slots = self.num_active_jobs - len(self.active)
        if self._closed or self.finished or slots < self.min_open_slots:
            return 0
        remaining = self.num_updates * self.buffer_size - self.accepted_count - len(self.active)
        return min(slots, remaining)

    def dispatch(self, available_clients: list[str]) -> list[Assignment]:
        """Reserve physical clients in order, sharing the current immutable state."""
        if len(set(available_clients)) != len(available_clients):
            raise ValueError("available client names must be unique")
        busy = {assignment.client_name for assignment in self.active.values()}
        clients = [name for name in available_clients if name not in busy]
        assignments = []
        for name in clients[: self.available_slots]:
            assignment = Assignment(self._next_id, name, name, self.version, self.model)
            self._next_id += 1
            self.active[assignment.assignment_id] = assignment
            assignments.append(assignment)
        return assignments

    def accept(self, assignment: Assignment, result: AsyncResult | None, error: Any = None) -> bool:
        """Accept exactly once; return True only when a global version is published.

        Validate before changing state. Rejected input raises and ends the run;
        it never contributes to B. Equal weights are default; configured metric
        weights must be finite and strictly positive.
        """
        current = self.active.get(assignment.assignment_id)
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
        base_model = assignment.base_model
        zero_delta = self.adapter.difference(base_model, base_model)
        reference = zero_delta if result.is_delta else base_model
        self.adapter.validate(result.model, reference)
        delta = result.model if result.is_delta else self.adapter.difference(result.model, base_model)
        weight = 1.0 if self.weight_metric is None else float(result.metrics[self.weight_metric])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("aggregation weight must be finite and positive")
        update = None
        if self.update_transform is not None or self.update_observer is not None:
            # Match the usual FULL/DIFF aggregation path: a delta can be
            # transformed/scaled without first constructing its full model.
            # Bind the original delta before the transform replaces it below.
            returned_model = (
                (lambda base=base_model, client_delta=delta: self.adapter.apply(base, client_delta, 1.0))
                if result.is_delta
                else result.model
            )
            update = AcceptedUpdate(
                assignment, self.version, self.model, delta, weight, result.metrics, returned_model, result.metadata
            )
        if self.update_transform is not None:
            delta = self.update_transform(update)
        self.adapter.validate(delta, zero_delta)
        if self.update_observer is not None:
            self.update_observer(update)
        self._buffer.add(delta, weight, str(assignment.assignment_id), assignment.base_version)
        del self.active[assignment.assignment_id]
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

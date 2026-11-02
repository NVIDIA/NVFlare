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

"""Buffered async training: recipe, Collab workflow, and training coordinator."""

import math
import queue
import threading
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Protocol

from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.collab import collab
from nvflare.collab.api import ContextKey
from nvflare.collab.recipe import CollabRecipe


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
    when the coordinator explicitly configures a weight_metric.
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


def _validate_options(
    num_updates, num_active_jobs, buffer_size, min_open_slots, server_lr, weight_metric, update_transform
):
    """Share early recipe validation with standalone coordinator construction."""
    for name, value in (
        ("num_updates", num_updates),
        ("num_active_jobs", num_active_jobs),
        ("buffer_size", buffer_size),
        ("min_open_slots", min_open_slots),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if min_open_slots > num_active_jobs:
        raise ValueError("min_open_slots cannot exceed num_active_jobs")
    if not math.isfinite(server_lr) or server_lr <= 0:
        raise ValueError("server_lr must be finite and positive")
    if weight_metric is not None and (not isinstance(weight_metric, str) or not weight_metric):
        raise ValueError("weight_metric must be a nonempty string or None")
    if update_transform is not None and not callable(update_transform):
        raise TypeError("update_transform must be callable")


class AsyncTrainingCoordinator:
    """K active assignments, B accepted updates/version, O slots before refill.

    Single-owner: invoke dispatch/accept on the workflow thread.
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
        _validate_options(
            num_updates, num_active_jobs, buffer_size, min_open_slots, server_lr, weight_metric, update_transform
        )
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
        if self._closed or current is None:
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


class AsyncWorkflow:
    """Accept actual arrivals, retaining assignment bases until acceptance.

    application.initialize() returns the initial model. Optional on_model,
    on_update, on_event, on_finish and close hooks own application artifacts.
    Hooks and update_transform run on the workflow thread; borrowed model
    states must not be mutated. The published client protocol is
    train(assignment_ids, model_version, global_model, logical_assignments),
    returning (model, metrics, metadata). No retry or remote cancellation is
    provided. See docs/user_guide/data_scientist_guide/async_recipe.rst for deployment and ownership.
    """

    _coordinator_class = AsyncTrainingCoordinator

    def __init__(
        self,
        application,
        model_adapter,
        num_updates: int,
        num_active_jobs: int,
        buffer_size: int = 1,
        min_open_slots: int = 1,
        server_lr: float = 1.0,
        weight_metric: str | None = None,
        update_transform=None,
        call_timeout: float = 3600.0,
    ):
        _validate_options(
            num_updates, num_active_jobs, buffer_size, min_open_slots, server_lr, weight_metric, update_transform
        )
        if not math.isfinite(call_timeout) or call_timeout <= 0:
            raise ValueError("call_timeout must be finite and positive")
        if not callable(getattr(application, "initialize", None)):
            raise TypeError("application must provide initialize()")
        self.application = application
        self.model_adapter = model_adapter
        self.num_updates = num_updates
        self.num_active_jobs = num_active_jobs
        self.buffer_size = buffer_size
        self.min_open_slots = min_open_slots
        self.server_lr = server_lr
        self.weight_metric = weight_metric
        self.update_transform = update_transform
        self.call_timeout = call_timeout

    def _hook(self, name, *args):
        hook = getattr(self.application, name, None)
        if hook is not None:
            hook(*args)

    @staticmethod
    def _normalize(result, assignment):
        if result is None:
            return result
        model, metrics, metadata = result
        if metadata["logical_name"] != assignment.logical_name:
            raise ValueError("logical client metadata mismatch")
        return AsyncResult(
            metadata["assignment_id"],
            metadata["model_version"],
            model,
            metrics,
            is_delta=metadata.get("is_delta", False),
            metadata=metadata,
        )

    def _create_coordinator(self, initial):
        return self._coordinator_class(
            initial,
            self.model_adapter,
            self.num_updates,
            self.num_active_jobs,
            self.buffer_size,
            self.min_open_slots,
            self.server_lr,
            self.weight_metric,
            self.update_transform,
            update_observer=getattr(self.application, "on_update", None),
        )

    def _emit(self, phase, assignment, version, client_name=None):
        self._hook(
            "on_event",
            AsyncEvent(
                phase,
                assignment.assignment_id,
                assignment.logical_name,
                client_name or assignment.client_name,
                assignment.base_version,
                version,
                deepcopy(assignment.metadata),
            ),
        )

    @contextmanager
    def _calls(self, names):
        """Deliver replies directly to one bounded server queue.

        Callback admission defines receipt order. Per-call Collab handles hold
        only completion/error status; no result-reader threads stage payloads.
        The protected call/coordinator/run methods share transport and lifecycle
        with the research example, without a scheduler/storage plugin API.
        """
        stopped = False
        outcome_lock = threading.Lock()
        outcomes = queue.Queue(maxsize=len(names) or 1)
        pending, queued = {}, set()

        def submit(client_name, assignment, **kwargs):
            identity = assignment.assignment_id
            with outcome_lock:
                pending[identity] = (client_name, None)

            def receive_result(group_call_context, result):
                with outcome_lock:
                    if (
                        not stopped
                        and identity in pending
                        and identity not in queued
                        and not group_call_context.waiter.results.failures
                    ):
                        queued.add(identity)
                        outcomes.put_nowait((client_name, identity, result, None))
                # Collab's per-call completion record must not retain the model.
                return None

            results = collab.get_clients([client_name])(
                blocking=False, timeout=self.call_timeout, process_resp_cb=receive_result
            ).train(
                {client_name: identity},
                assignment.base_version,
                assignment.base_model,
                {client_name: assignment.logical_name},
                **kwargs,
            )
            with outcome_lock:
                pending[identity] = (client_name, results)

        def receive(timeout=0.1):
            # Collab invokes response callbacks only for successful replies.
            # Observe failures/timeouts without another queue or reader pool.
            with outcome_lock:
                for identity, (client_name, results) in pending.items():
                    if identity not in queued and results is not None:
                        error = results.failures.get(client_name)
                        if error is not None:
                            queued.add(identity)
                            outcomes.put_nowait((client_name, identity, None, error))
            outcome = outcomes.get(timeout=timeout)
            with outcome_lock:
                pending.pop(outcome[1])
                queued.remove(outcome[1])
            return outcome

        try:
            yield submit, receive
        finally:
            with outcome_lock:
                stopped = True
                pending.clear()
                queued.clear()
                while not outcomes.empty():
                    outcomes.get_nowait()
            # Existing Collab calls finish or time out; late replies are ignored.

    def _run(self, coordinator, names, submit, receive):
        if len(names) < self.num_active_jobs:
            raise ValueError("num_active_jobs exceeds available clients")
        available = list(names)

        def dispatch():
            for assignment in coordinator.dispatch(available):
                available.remove(assignment.client_name)
                submit(assignment.client_name, assignment)
                self._emit("dispatch", assignment, coordinator.version)

        while not coordinator.finished:
            if collab.is_aborted:
                raise RuntimeError("async workflow aborted")
            dispatch()
            try:
                client_name, identity, result, error = receive()
            except queue.Empty:
                continue
            assignment = coordinator.active[identity]
            version = coordinator.version
            published = coordinator.accept(assignment, self._normalize(result, assignment), error=error)
            available.append(client_name)
            self._emit("accept", assignment, version)
            if published:
                dispatch()  # Refill before application evaluation/artifact work.
                self._hook("on_model", coordinator.model, coordinator.version)

    @collab.main
    def execute(self):
        names = sorted(client.name for client in collab.clients)
        with ExitStack() as cleanup:
            cleanup.callback(self._hook, "close")
            transform_close = getattr(self.update_transform, "close", None)
            if transform_close is not None:
                cleanup.callback(transform_close)
            initial = self.application.initialize()
            initial = collab.get_prop(ContextKey.RESULT, initial)
            transform_init = getattr(self.update_transform, "initialize", None)
            if transform_init is not None:
                transform_init()
            coordinator = self._create_coordinator(initial)
            cleanup.callback(coordinator.close)
            self._hook("on_model", coordinator.model, coordinator.version)
            with self._calls(names) as (submit, receive):
                self._run(coordinator, names, submit, receive)
            self._hook("on_finish", coordinator.model, coordinator.version)
            return coordinator.model


class AsyncRecipe(CollabRecipe):
    """Actual-arrival Collab training: K active jobs, B updates, O refill slots.

    K=B=O gives a synchronous boundary. application supplies model/artifact
    hooks; model_adapter supplies framework-specific arithmetic. An optional
    update_transform modifies accepted deltas. No replay, seed policy or storage
    configuration is part of this recipe. Other options pass to CollabRecipe;
    min_clients defaults to K. Components must be reconstructible for export.
    """

    def __init__(
        self,
        job_name: str,
        client,
        application,
        model_adapter,
        num_updates: int,
        num_active_jobs: int,
        buffer_size: int = 1,
        min_open_slots: int = 1,
        server_lr: float = 1.0,
        weight_metric: str | None = None,
        update_transform=None,
        call_timeout: float = 3600.0,
        **collab_options,
    ):
        workflow = AsyncWorkflow(
            application,
            model_adapter,
            num_updates,
            num_active_jobs,
            buffer_size,
            min_open_slots,
            server_lr,
            weight_metric,
            update_transform,
            call_timeout,
        )
        collab_options.setdefault("min_clients", num_active_jobs)
        super().__init__(job_name=job_name, server=workflow, client=client, **collab_options)

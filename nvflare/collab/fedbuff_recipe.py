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

"""FedBuff training: recipe, Collab workflow, and training coordinator."""

import math
import queue
import threading
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable

from nvflare.app_common.aggregators.weighted_aggregation_helper import WeightedAggregationHelper
from nvflare.collab import collab
from nvflare.collab.api import ContextKey
from nvflare.collab.model_ops import ModelOps
from nvflare.collab.recipe import CollabRecipe
from nvflare.collab.replayable_execution import AsyncExecutionPort, run_async_execution


@dataclass(frozen=True)
class Assignment:
    """Server-local identity and an immutable model reference until acceptance.

    Model operations return a new state for each publication, so assignments from
    the same version safely share a base without copying or reference counts.
    """

    assignment_id: int
    client_name: str
    logical_name: str
    base_version: int
    base_model: dict = field(repr=False)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FedBuffEvent:
    """Lightweight observer trace; no model states are retained."""

    phase: str
    assignment_id: int
    logical_name: str
    client_name: str
    base_version: int
    current_version: int
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FedBuffResult:
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
        uses the model operations' original dtype/buffer rules and can fail if the
        unscaled full model is unrepresentable, even when a scaled update is
        valid. This does not affect acceptance unless a consumer requests it.
        """
        return self._returned_model() if callable(self._returned_model) else self._returned_model

    @property
    def staleness(self):
        return self.current_version - self.assignment.base_version


def _validate_options(
    num_updates, concurrency, buffer_size, min_open_slots, server_lr, weight_metric, update_transform
):
    """Share early recipe validation with standalone coordinator construction."""
    for name, value in (
        ("num_updates", num_updates),
        ("concurrency", concurrency),
        ("buffer_size", buffer_size),
        ("min_open_slots", min_open_slots),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if min_open_slots > concurrency:
        raise ValueError("min_open_slots cannot exceed concurrency")
    if not math.isfinite(server_lr) or server_lr <= 0:
        raise ValueError("server_lr must be finite and positive")
    if weight_metric is not None and (not isinstance(weight_metric, str) or not weight_metric):
        raise ValueError("weight_metric must be a nonempty string or None")
    if update_transform is not None and not callable(update_transform):
        raise TypeError("update_transform must be callable")


class FedBuffTrainingCoordinator:
    """Track active clients, buffered contributions and the refill threshold.

    Single-owner: invoke dispatch/accept on the workflow thread.
    Stops after num_updates global publications, issuing no speculative work
    beyond that accepted-contribution budget. Errors are fail-fast; there is no
    automatic retry. Duplicate/stale terminal events are ignored by assignment
    identity. close() discards partial accumulation and releases base snapshots.
    """

    def __init__(
        self,
        initial_model: dict,
        model_ops: ModelOps,
        num_updates: int,
        concurrency: int,
        buffer_size: int = 1,
        min_open_slots: int = 1,
        server_lr: float = 1.0,
        weight_metric: str | None = None,
        update_transform=None,
        update_observer=None,
    ):
        _validate_options(
            num_updates, concurrency, buffer_size, min_open_slots, server_lr, weight_metric, update_transform
        )
        self.model_ops = model_ops
        self.model = model_ops.snapshot(initial_model)
        model_ops.validate(self.model, self.model)
        self.num_updates = num_updates
        self.concurrency = concurrency
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
        """Capacity allowed by active-client/refill limits and the remaining contribution budget."""
        slots = self.concurrency - len(self.active)
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
        return self.reserve(clients)

    def reserve(
        self, logical_names: list[str], metadata: list[dict] | None = None, assignment_factory=None
    ) -> list[Assignment]:
        """Bind logical work to the current model before a physical worker is chosen."""
        if len(set(logical_names)) != len(logical_names):
            raise ValueError("logical client names must be unique within a reservation")
        if metadata is not None and len(metadata) != len(logical_names):
            raise ValueError("metadata must match logical client names")
        assignments = []
        for index, name in enumerate(logical_names[: self.available_slots]):
            details = deepcopy(metadata[index]) if metadata is not None else {}
            assignment = (
                assignment_factory(self._next_id, name, self.version, self.model, details)
                if assignment_factory is not None
                else Assignment(self._next_id, name, name, self.version, self.model, details)
            )
            self._next_id += 1
            self.active[assignment.assignment_id] = assignment
            assignments.append(assignment)
        return assignments

    def accept(self, assignment: Assignment, result: FedBuffResult | None, error: Any = None) -> bool:
        """Accept exactly once; return True only when a global version is published.

        Validate before changing state. Rejected input raises and ends the run;
        it never contributes to the aggregation buffer. Equal weights are default; configured metric
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
        if not isinstance(result, FedBuffResult):
            raise TypeError("result must be FedBuffResult")
        if result.assignment_id != assignment.assignment_id or result.base_version != assignment.base_version:
            raise ValueError("assignment metadata mismatch")
        base_model = assignment.base_model
        zero_delta = self.model_ops.difference(base_model, base_model)
        reference = zero_delta if result.is_delta else base_model
        self.model_ops.validate(result.model, reference)
        delta = result.model if result.is_delta else self.model_ops.difference(result.model, base_model)
        weight = 1.0 if self.weight_metric is None else float(result.metrics[self.weight_metric])
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("aggregation weight must be finite and positive")
        update = None
        if self.update_transform is not None or self.update_observer is not None:
            # Match the usual FULL/DIFF aggregation path: a delta can be
            # transformed/scaled without first constructing its full model.
            # Bind the original delta before the transform replaces it below.
            returned_model = (
                (lambda base=base_model, client_delta=delta: self.model_ops.apply(base, client_delta, 1.0))
                if result.is_delta
                else result.model
            )
            update = AcceptedUpdate(
                assignment, self.version, self.model, delta, weight, result.metrics, returned_model, result.metadata
            )
        if self.update_transform is not None:
            delta = self.update_transform(update)
        self.model_ops.validate(delta, zero_delta)
        if self.update_observer is not None:
            self.update_observer(update)
        self._buffer.add(delta, weight, str(assignment.assignment_id), assignment.base_version)
        del self.active[assignment.assignment_id]
        self.accepted_count += 1
        if self.buffered_count < self.buffer_size:
            return False
        self.model = self.model_ops.apply(self.model, self._buffer.get_result(), self.server_lr)
        self.version += 1
        return True

    def close(self):
        """Release outstanding bases and partial accumulation; do not publish a partial version."""
        self._closed = True
        self.active.clear()
        self._buffer.reset_stats()


class FedBuffWorkflow:
    """Accept actual arrivals, retaining assignment bases until acceptance.

    model_loader.load_model() returns the initial state. The optional observer may implement
    initialize, on_model, on_update, on_event, on_finish and close for artifacts.
    Observer hooks and update_transform run on the workflow thread; borrowed model
    states must not be mutated. The published client protocol is
    train(assignment_ids, model_version, global_model, logical_assignments),
    returning (model, metrics, metadata). No retry or remote cancellation is
    provided. See docs/user_guide/data_scientist_guide/fedbuff_recipe.rst for deployment and ownership.
    """

    _coordinator_class = FedBuffTrainingCoordinator

    def __init__(
        self,
        model_loader,
        model_ops,
        num_updates: int,
        concurrency: int,
        buffer_size: int = 1,
        min_open_slots: int = 1,
        server_lr: float = 1.0,
        weight_metric: str | None = None,
        update_transform=None,
        call_timeout: float = 3600.0,
        observer=None,
    ):
        _validate_options(
            num_updates, concurrency, buffer_size, min_open_slots, server_lr, weight_metric, update_transform
        )
        if not math.isfinite(call_timeout) or call_timeout <= 0:
            raise ValueError("call_timeout must be finite and positive")
        if not callable(getattr(model_loader, "load_model", None)):
            raise TypeError("model_loader must provide load_model()")
        self.model_loader = model_loader
        self.model_ops = model_ops
        self.observer = observer
        self.num_updates = num_updates
        self.concurrency = concurrency
        self.buffer_size = buffer_size
        self.min_open_slots = min_open_slots
        self.server_lr = server_lr
        self.weight_metric = weight_metric
        self.update_transform = update_transform
        self.call_timeout = call_timeout

    def _notify(self, name, *args):
        hook = getattr(self.observer, name, None)
        if hook is not None:
            hook(*args)

    @staticmethod
    def _normalize(result, assignment):
        if result is None:
            return result
        model, metrics, metadata = result
        if metadata["logical_name"] != assignment.logical_name:
            raise ValueError("logical client metadata mismatch")
        return FedBuffResult(
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
            self.model_ops,
            self.num_updates,
            self.concurrency,
            self.buffer_size,
            self.min_open_slots,
            self.server_lr,
            self.weight_metric,
            self.update_transform,
            update_observer=getattr(self.observer, "on_update", None),
        )

    def _emit(self, phase, assignment, version, client_name=None):
        self._notify(
            "on_event",
            FedBuffEvent(
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
        if len(names) < self.concurrency:
            raise ValueError("concurrency exceeds available clients")
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
                dispatch()  # Refill before observer evaluation/artifact work.
                self._notify("on_model", coordinator.model, coordinator.version)

    def _execution_port(self, coordinator, names, submit, receive):
        """Expose algorithm decisions to the shared live/planned execution policy."""
        return AsyncExecutionPort(
            state=coordinator,
            clients=names,
            contribution_budget=coordinator.num_updates * coordinator.buffer_size,
            submit=submit,
            receive=receive,
            live_run=lambda: self._run(coordinator, names, submit, receive),
            normalize=self._normalize,
            on_dispatch=lambda assignment, version, site: self._emit("dispatch", assignment, version, site),
            on_accept=lambda assignment, version, site: self._emit("accept", assignment, version, site),
            on_publish=lambda model, version: self._notify("on_model", model, version),
            on_time=lambda instant: self._notify("on_time", instant),
            aborted=lambda: collab.is_aborted,
        )

    @collab.main
    def execute(self):
        names = sorted(client.name for client in collab.clients)
        with ExitStack() as cleanup:
            observer_close = getattr(self.observer, "close", None)
            if observer_close is not None:
                cleanup.callback(observer_close)
            observer_init = getattr(self.observer, "initialize", None)
            if observer_init is not None:
                observer_init()
            loader_close = getattr(self.model_loader, "close", None)
            if loader_close is not None:
                cleanup.callback(loader_close)
            transform_close = getattr(self.update_transform, "close", None)
            if transform_close is not None:
                cleanup.callback(transform_close)
            initial = self.model_loader.load_model()
            initial = collab.get_prop(ContextKey.RESULT, initial)
            transform_init = getattr(self.update_transform, "initialize", None)
            if transform_init is not None:
                transform_init()
            coordinator = self._create_coordinator(initial)
            cleanup.callback(coordinator.close)
            self._notify("on_model", coordinator.model, coordinator.version)
            with self._calls(names) as (submit, receive):
                run_async_execution(self._execution_port(coordinator, names, submit, receive), collab.fl_ctx)
            self._notify("on_finish", coordinator.model, coordinator.version)
            return coordinator.model


class FedBuffRecipe(CollabRecipe):
    """FedBuff-style buffered aggregation in actual client arrival order.

    concurrency limits concurrent client assignments; buffer_size counts
    accepted contributions per publication; min_open_slots sets the refill
    threshold. A threshold of one refills each completed slot immediately.
    Setting all three counts equal gives a synchronous training boundary.

    model_loader supplies the initial model; model_ops supplies arithmetic.
    An optional observer owns evaluation and artifacts.
    An optional update_transform modifies accepted deltas. Simulation plans
    are supplied by SimEnv, not by the exported recipe. Other options pass to CollabRecipe;
    min_clients defaults to concurrency. Components must be reconstructible.
    """

    def __init__(
        self,
        job_name: str,
        client,
        model_loader,
        model_ops,
        num_updates: int,
        concurrency: int,
        buffer_size: int = 1,
        min_open_slots: int = 1,
        server_lr: float = 1.0,
        weight_metric: str | None = None,
        update_transform=None,
        call_timeout: float = 3600.0,
        observer=None,
        **collab_options,
    ):
        workflow = FedBuffWorkflow(
            model_loader,
            model_ops,
            num_updates,
            concurrency,
            buffer_size,
            min_open_slots,
            server_lr,
            weight_metric,
            update_transform,
            call_timeout,
            observer,
        )
        collab_options.setdefault("min_clients", concurrency)
        super().__init__(job_name=job_name, server=workflow, client=client, **collab_options)

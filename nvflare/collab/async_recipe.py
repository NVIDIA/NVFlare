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

"""Buffered actual-arrival training over the existing Collab runtime."""

import math
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from copy import deepcopy

from nvflare.collab import collab
from nvflare.collab.api import ContextKey
from nvflare.collab.async_engine import AsyncEngine, AsyncEvent, AsyncResult, _validate_options
from nvflare.collab.recipe import CollabRecipe


class AsyncWorkflow:
    """Accept actual arrivals, retaining assignment bases until acceptance.

    application.initialize() returns the initial model. Optional on_model,
    on_update, on_event, on_finish and close hooks own application artifacts.
    Hooks and update_transform run on the workflow thread; borrowed model
    states must not be mutated. The published client protocol is
    train(assignment_ids, model_version, global_model, logical_assignments),
    returning (model, metrics, metadata). No retry or remote cancellation is
    provided. See docs/design/async_recipe.md for deployment and ownership.
    """

    _engine_class = AsyncEngine

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

    def _create_engine(self, initial):
        return self._engine_class(
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
        """Bound result readers and queued payloads by the physical worker pool.

        The protected call/engine/run methods let an example own additional
        orchestration while reusing Collab transport, cleanup and aggregation.
        They are implementation hooks, not a scheduler/storage plugin API.
        """
        stopped = False
        outcome_lock = threading.Lock()
        outcomes = queue.Queue(maxsize=len(names) or 1)
        pool = ThreadPoolExecutor(max_workers=max(1, len(names)), thread_name_prefix="async_results")

        def watch(client_name, assignment_id, results):
            result, error = None, None
            try:
                for site, value in results:
                    if site != client_name:
                        raise ValueError("unexpected result source")
                    result = value
                error = results.failures.get(client_name)
            except Exception as exc:
                error = exc
            with outcome_lock:
                if not stopped:
                    # One outstanding call per worker bounds queued outcomes.
                    outcomes.put_nowait((client_name, assignment_id, result, error))

        def submit(client_name, assignment, **kwargs):
            results = collab.get_clients([client_name])(blocking=False, timeout=self.call_timeout).train(
                {client_name: assignment.assignment_id},
                assignment.base_version,
                assignment.base_model,
                {client_name: assignment.logical_name},
                **kwargs,
            )
            pool.submit(watch, client_name, assignment.assignment_id, results)

        try:
            yield submit, outcomes
        finally:
            with outcome_lock:
                stopped = True
                while not outcomes.empty():
                    outcomes.get_nowait()
            # Drain existing ResultQueues to their Collab timeout. Local cleanup
            # cannot cancel remote training; readers discard subsequent replies.
            pool.shutdown(wait=False, cancel_futures=False)

    def _run(self, engine, names, submit, outcomes):
        if len(names) < self.num_active_jobs:
            raise ValueError("num_active_jobs exceeds available clients")
        available = list(names)

        def dispatch():
            for assignment in engine.dispatch(available):
                available.remove(assignment.client_name)
                submit(assignment.client_name, assignment)
                self._emit("dispatch", assignment, engine.version)

        while not engine.finished:
            if collab.is_aborted:
                raise RuntimeError("async workflow aborted")
            dispatch()
            try:
                client_name, identity, result, error = outcomes.get(timeout=0.1)
            except queue.Empty:
                continue
            assignment = engine.active[identity]
            version = engine.version
            published = engine.accept(assignment, self._normalize(result, assignment), error=error)
            available.append(client_name)
            self._emit("accept", assignment, version)
            if published:
                dispatch()  # Refill before application evaluation/artifact work.
                self._hook("on_model", engine.model, engine.version)

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
            engine = self._create_engine(initial)
            cleanup.callback(engine.close)
            self._hook("on_model", engine.model, engine.version)
            with self._calls(names) as (submit, outcomes):
                self._run(engine, names, submit, outcomes)
            self._hook("on_finish", engine.model, engine.version)
            return engine.model


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

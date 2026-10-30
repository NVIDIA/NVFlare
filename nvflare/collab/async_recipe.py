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
    provided. See docs/user_guide/data_scientist_guide/async_recipe.rst for deployment and ownership.
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
        """Deliver replies directly to one bounded server queue.

        Callback admission defines receipt order. Per-call Collab handles hold
        only completion/error status; no result-reader threads stage payloads.
        The protected call/engine/run methods share transport and lifecycle
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

    def _run(self, engine, names, submit, receive):
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
                client_name, identity, result, error = receive()
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
            with self._calls(names) as (submit, receive):
                self._run(engine, names, submit, receive)
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

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

"""Collab adapter and recipe for the standalone asynchronous engine."""

import math
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy

from nvflare.collab import collab
from nvflare.collab.api import ContextKey
from nvflare.collab.async_scheduling import ArrivalOrderScheduler, AsyncEvent
from nvflare.collab.async_storage import MemoryResultStore, MemorySnapshotStore
from nvflare.collab.async_workflow import AsyncEngine, AsyncResult, _positive_int
from nvflare.collab.recipe import CollabRecipe


class AsyncWorkflow:
    """Execute scheduled asynchronous training using the published train protocol.

    application.initialize() returns the initial model. Optional hooks are
    on_model(model, version) (initial and each publication),
    on_finish(model, version) (after completion), on_update(update),
    on_event(event), and close() (always).
    Hooks and update_transform run on the main workflow thread. They must not
    mutate borrowed model states. The application owns evaluation/artifacts.

    Clients publish train(assignment_ids, model_version, global_model,
    logical_assignments). They return a tuple
    (model, metrics, metadata), with assignment_id/model_version/logical_name.
    Scheduling, extra train arguments, and model storage are replaceable.
    No automatic retry or remote cancellation is claimed. A timed-out call can
    still be running remotely; failure stops the workflow and closes admission.
    """

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
        scheduler=None,
        train_kwargs=None,
        snapshot_store=None,
        result_store=None,
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
        if not math.isfinite(call_timeout) or call_timeout <= 0:
            raise ValueError("call_timeout must be finite and positive")
        if not math.isfinite(server_lr) or server_lr <= 0:
            raise ValueError("server_lr must be finite and positive")
        if weight_metric is not None and (not isinstance(weight_metric, str) or not weight_metric):
            raise ValueError("weight_metric must be a nonempty string or None")
        if update_transform is not None and not callable(update_transform):
            raise TypeError("update_transform must be callable")
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
        if train_kwargs is not None and not callable(train_kwargs):
            raise TypeError("train_kwargs must be callable")
        self.call_timeout = call_timeout
        self.scheduler = scheduler
        self.train_kwargs = train_kwargs
        self.snapshot_store = snapshot_store
        self.result_store = result_store

    def _hook(self, name, *args):
        hook = getattr(self.application, name, None)
        if hook is not None:
            hook(*args)

    @staticmethod
    def _normalize(result, assignment):
        if result is None or isinstance(result, AsyncResult):
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

    @collab.main
    def execute(self):
        stopped = threading.Event()
        outcome_lock = threading.Lock()
        names = sorted(client.name for client in collab.clients)
        outcomes = queue.Queue(maxsize=len(names) or 1)
        scheduler = self.scheduler if self.scheduler is not None else ArrivalOrderScheduler()
        result_store = self.result_store if self.result_store is not None else MemoryResultStore()
        snapshot_store = self.snapshot_store if self.snapshot_store is not None else MemorySnapshotStore()

        def optional(target, name, *args):
            hook = getattr(target, name, None)
            if hook is not None:
                return hook(*args)

        def watch(call, results):
            result = None
            error = None
            transferred = False
            try:
                try:
                    for site, value in results:
                        if site != call.client_name:
                            raise ValueError("unexpected result source")
                        result = value
                    error = results.failures.get(call.client_name)
                except Exception as exc:
                    error = exc
                while not stopped.is_set():
                    with outcome_lock:
                        if stopped.is_set():
                            return
                        try:
                            outcomes.put_nowait((call, result, error))
                            transferred = True
                            return
                        except queue.Full:
                            pass
                    stopped.wait(0.1)
            finally:
                if not transferred:
                    result_store.release(result)

        with ExitStack() as cleanup:
            cleanup.callback(self._hook, "close")
            cleanup.callback(optional, self.update_transform, "close")
            cleanup.callback(scheduler.close)
            cleanup.callback(result_store.close)
            cleanup.callback(snapshot_store.close)
            initial = self.application.initialize()
            initial = collab.get_prop(ContextKey.RESULT, initial)
            optional(self.update_transform, "initialize")
            optional(snapshot_store, "initialize")
            optional(result_store, "initialize")
            engine = AsyncEngine(
                initial,
                self.model_adapter,
                self.num_updates,
                self.num_active_jobs,
                self.buffer_size,
                self.min_open_slots,
                self.server_lr,
                self.weight_metric,
                self.update_transform,
                snapshot_store,
                update_observer=getattr(self.application, "on_update", None),
            )
            cleanup.callback(engine.close)
            scheduler.initialize(engine, names)
            pool = ThreadPoolExecutor(max_workers=max(1, len(names)), thread_name_prefix="async_results")

            def stop_readers():
                with outcome_lock:
                    stopped.set()
                    while not outcomes.empty():
                        _, result, _ = outcomes.get_nowait()
                        result_store.release(result)
                # Queued readers still drain their ResultQueues and release late
                # offloaded payloads. Cancellation would bypass that cleanup.
                pool.shutdown(wait=False, cancel_futures=False)

            cleanup.callback(stop_readers)
            # A custom scheduler may hold completed results. Keep physical
            # calls separate from assignments still awaiting acceptance.
            inflight = {}
            completed = {}
            submitted = set()

            def emit(phase, call, version):
                assignment = call.assignment
                self._hook(
                    "on_event",
                    AsyncEvent(
                        phase,
                        assignment.assignment_id,
                        assignment.logical_name,
                        call.client_name,
                        assignment.base_version,
                        version,
                        deepcopy(assignment.metadata),
                        getattr(scheduler, "time", None),
                    ),
                )

            def dispatch_available():
                if engine.finished or scheduler.finished:
                    return
                for call in scheduler.dispatch():
                    assignment = call.assignment
                    if call.client_name not in names or call.client_name in inflight:
                        raise ValueError("scheduler selected an unavailable physical worker")
                    if (
                        engine.active.get(assignment.assignment_id) is not assignment
                        or assignment.assignment_id in submitted
                    ):
                        raise ValueError("scheduler selected an unreserved or already submitted assignment")
                    kwargs = self.train_kwargs(call) if self.train_kwargs is not None else {}
                    if not isinstance(kwargs, dict):
                        raise TypeError("train_kwargs must return a dictionary")
                    results = collab.get_clients([call.client_name])(blocking=False, timeout=self.call_timeout).train(
                        {call.client_name: assignment.assignment_id},
                        assignment.base_version,
                        assignment.base_model,
                        {call.client_name: assignment.logical_name},
                        **kwargs,
                    )
                    inflight[call.client_name] = call
                    submitted.add(assignment.assignment_id)
                    pool.submit(watch, call, results)
                    emit("dispatch", call, engine.version)

            self._hook("on_model", engine.model, engine.version)
            while not engine.finished and not scheduler.finished:
                if collab.is_aborted:
                    raise RuntimeError("async workflow aborted")
                dispatch_available()
                assignment_id = scheduler.next_result()
                if scheduler.finished:
                    break
                if assignment_id is not None:
                    call = completed.pop(assignment_id, None)
                    if call is None:
                        raise ValueError("scheduler selected an incomplete result")
                    result = result_store.take(assignment_id)
                    acceptance_version = engine.version
                    try:
                        published = engine.accept(call.assignment, self._normalize(result, call.assignment))
                    finally:
                        result_store.release(result)
                    submitted.remove(assignment_id)
                    scheduler.on_accept(call.assignment)
                    emit("accept", call, acceptance_version)
                    if published:
                        # Preserve publication/refill semantics before evaluation.
                        dispatch_available()
                        self._hook("on_model", engine.model, engine.version)
                    continue
                # A scheduling decision can unlock work without yielding a
                # result. Retry dispatch before declaring that progress stopped.
                dispatch_available()
                if not inflight:
                    raise RuntimeError("async workflow cannot make progress")
                try:
                    call, result, error = outcomes.get(timeout=0.1)
                except queue.Empty:
                    continue
                inflight.pop(call.client_name)
                if error is not None or result is None:
                    result_store.release(result)
                    raise RuntimeError(f"assignment {call.assignment.assignment_id} failed: {error or 'no result'}")
                try:
                    result_store.put(call.assignment.assignment_id, result)
                except Exception:
                    result_store.release(result)
                    raise
                completed[call.assignment.assignment_id] = call
                scheduler.on_result(call)
            self._hook("on_finish", engine.model, engine.version)
            return engine.model


class AsyncRecipe(CollabRecipe):
    """Buffered asynchronous training over CollabAPI.

    K=num_active_jobs, B=buffer_size, O=min_open_slots; K=B=O gives a
    synchronous boundary. Defaults accept actual arrivals, with one call per
    physical client. application.initialize() supplies the initial model;
    optional on_model/on_update/on_event/on_finish/close hooks own artifacts.

    model_adapter owns arithmetic. update_transform is an optional callable
    receiving AcceptedUpdate and returning a replacement delta; optional
    initialize()/close() hooks manage its resources. train_kwargs(call) can
    supply extra client train arguments. scheduler, snapshot_store and
    result_store accept components implementing the corresponding protocols;
    defaults use arrival order and bounded in-memory storage.

    Components must be importable and reconstructible for job export. All
    hooks run on the workflow thread except late-result store release during
    cleanup. Other options pass to CollabRecipe; min_clients defaults to the
    scheduler's required_workers(K), or K. See docs/design/async_recipe.md.
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
        scheduler=None,
        train_kwargs=None,
        snapshot_store=None,
        result_store=None,
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
            scheduler,
            train_kwargs,
            snapshot_store,
            result_store,
        )
        collab_options.setdefault(
            "min_clients", num_active_jobs if scheduler is None else scheduler.required_workers(num_active_jobs)
        )
        super().__init__(job_name=job_name, server=workflow, client=client, **collab_options)

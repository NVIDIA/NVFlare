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

from nvflare.collab import collab
from nvflare.collab.api import ContextKey
from nvflare.collab.async_workflow import AsyncEngine, AsyncResult, _positive_int
from nvflare.collab.recipe import CollabRecipe


class AsyncWorkflow:
    """Execute actual-arrival training using the published train protocol.

    application.initialize() returns the initial model. Optional hooks are
    on_model(model, version) (initial and each publication),
    on_finish(model, version) (after completion), and close() (always).
    Hooks and update_transform run on the main workflow thread. They must not
    mutate borrowed model states. The application owns evaluation/artifacts.

    Clients publish train(assignment_ids, model_version, global_model,
    logical_assignments). They return a tuple
    (model, metrics, metadata), with assignment_id/model_version/logical_name.
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
        self.call_timeout = call_timeout

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
        )

    @collab.main
    def execute(self):
        engine = None
        pool = None
        stopped = threading.Event()
        outcome_lock = threading.Lock()
        outcomes = queue.Queue(maxsize=self.num_active_jobs)

        def watch(assignment, results):
            result = None
            error = None
            try:
                for site, value in results:
                    if site != assignment.client_name:
                        raise ValueError("unexpected result source")
                    result = value
                error = results.failures.get(assignment.client_name)
            except Exception as exc:
                error = exc
            while not stopped.is_set():
                with outcome_lock:
                    if stopped.is_set():
                        return
                    try:
                        outcomes.put_nowait((assignment, result, error))
                        return
                    except queue.Full:
                        pass
                stopped.wait(0.1)

        try:
            names = [client.name for client in collab.clients]
            if len(names) < self.num_active_jobs:
                raise ValueError("num_active_jobs exceeds available clients")
            initial = self.application.initialize()
            initial = collab.get_prop(ContextKey.RESULT, initial)
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
            )
            pool = ThreadPoolExecutor(max_workers=self.num_active_jobs, thread_name_prefix="async_results")
            self._hook("on_model", engine.model, engine.version)
            available = list(names)

            def dispatch_available():
                for assignment in engine.dispatch(available):
                    available.remove(assignment.client_name)
                    results = collab.get_clients([assignment.client_name])(
                        blocking=False, timeout=self.call_timeout
                    ).train(
                        {assignment.client_name: assignment.assignment_id},
                        assignment.base_version,
                        assignment.base_model,
                        {assignment.client_name: assignment.logical_name},
                    )
                    pool.submit(watch, assignment, results)

            while not engine.finished:
                if collab.is_aborted:
                    raise RuntimeError("async workflow aborted")
                dispatch_available()
                if not engine.active:
                    raise RuntimeError("async workflow cannot make progress")
                try:
                    assignment, result, error = outcomes.get(timeout=0.1)
                except queue.Empty:
                    continue
                published = engine.accept(assignment, self._normalize(result, assignment), error)
                if assignment.client_name not in engine.active and assignment.client_name not in available:
                    available.append(assignment.client_name)
                if published:
                    # Refill with the new version before artifact/evaluation work.
                    dispatch_available()
                    self._hook("on_model", engine.model, engine.version)
            self._hook("on_finish", engine.model, engine.version)
            return engine.model
        finally:
            with outcome_lock:
                stopped.set()
                while not outcomes.empty():
                    outcomes.get_nowait()
            if engine is not None:
                engine.close()
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            self._hook("close")


class AsyncRecipe(CollabRecipe):
    """Buffered asynchronous Collab training with explicit application/model adapters.

    K=num_active_jobs, B=buffer_size, O=min_open_slots. Use K=B=O for a
    synchronous boundary. No KD or tensor framework is imported by this recipe.
    Additional runtime options are forwarded to CollabRecipe.
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

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

"""Research-owned replay and disk bookkeeping over the shared async engine."""

import queue
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, field

from pt_async_cifar10_reproducible.disk_storage import DiskResultStore, DiskSnapshotStore

from nvflare.collab import collab
from nvflare.collab.async_recipe import AsyncWorkflow
from nvflare.collab.async_workflow import AsyncEngine
from nvflare.collab.recipe import CollabRecipe


@dataclass(frozen=True)
class DiskAssignment:
    """Expose the engine's assignment attributes with an on-demand disk base."""

    assignment_id: int
    client_name: str | None
    logical_name: str
    base_version: int
    base_path: str
    metadata: dict = field(default_factory=dict)

    @property
    def base_model(self):
        return DiskSnapshotStore._load(self.base_path)


class ResearchEngine(AsyncEngine):
    """Keep replay reservations on disk; inherit all acceptance/aggregation math."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshots = DiskSnapshotStore()
        self.snapshots.initialize()

    def dispatch(self, available_clients):
        if len(set(available_clients)) != len(available_clients):
            raise ValueError("available client names must be unique")
        busy = {assignment.client_name for assignment in self.active.values()}
        names = [name for name in available_clients if name not in busy]
        return self._reserve(names, names)

    def reserve(self, logical_names, metadata=None):
        if any(not isinstance(name, str) or not name for name in logical_names):
            raise ValueError("logical names must be nonempty strings")
        if metadata is not None and (
            len(metadata) != len(logical_names) or any(not isinstance(m, dict) for m in metadata)
        ):
            raise ValueError("metadata must contain a dictionary per logical assignment")
        return self._reserve(logical_names, [None] * len(logical_names), metadata)

    def _reserve(self, logical_names, client_names, metadata=None):
        assignments = []
        for index in range(min(len(logical_names), self.available_slots)):
            path = self.snapshots.retain(self.version, self.model, self.adapter)
            assignment = DiskAssignment(
                self._next_id,
                client_names[index],
                logical_names[index],
                self.version,
                path,
                deepcopy(metadata[index]) if metadata else {},
            )
            self._next_id += 1
            self.active[assignment.assignment_id] = assignment
            assignments.append(assignment)
        return assignments


class ResearchWorkflow(AsyncWorkflow):
    """Own fixed replay, optional train arguments, and retained disk results.

    Core supplies Collab calls, lifecycle and numerical acceptance. This example
    owns logical reservations, worker reuse, held results and scheduler horizons.
    """

    def __init__(
        self,
        application,
        model_adapter,
        num_updates: int,
        num_active_jobs: int,
        scheduler,
        train_kwargs=None,
        buffer_size: int = 1,
        min_open_slots: int = 1,
        server_lr: float = 1.0,
        weight_metric: str | None = None,
        update_transform=None,
        call_timeout: float = 3600.0,
    ):
        super().__init__(
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
        if train_kwargs is not None and not callable(train_kwargs):
            raise TypeError("train_kwargs must be callable")
        self.scheduler = scheduler
        self.train_kwargs = train_kwargs

    def _create_engine(self, initial):
        return ResearchEngine(
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

    def _run(self, engine, names, submit, outcomes):
        scheduler = self.scheduler
        with ExitStack() as cleanup:
            cleanup.callback(scheduler.close)
            results = DiskResultStore()
            results.initialize()
            scheduler.initialize(engine, names)
            inflight, completed, submitted = {}, {}, set()

            def dispatch():
                if engine.finished or scheduler.finished:
                    return
                for call in scheduler.dispatch():
                    assignment = call.assignment
                    if call.client_name not in names or call.client_name in inflight:
                        raise ValueError("scheduler selected an unavailable physical worker")
                    identity = assignment.assignment_id
                    if engine.active.get(identity) is not assignment or identity in submitted:
                        raise ValueError("scheduler selected an unreserved or already submitted assignment")
                    kwargs = self.train_kwargs(call) if self.train_kwargs is not None else {}
                    if not isinstance(kwargs, dict):
                        raise TypeError("train_kwargs must return a dictionary")
                    submit(call.client_name, assignment, **kwargs)
                    inflight[call.client_name] = call
                    submitted.add(identity)
                    self._emit("dispatch", assignment, engine.version, call.client_name)

            while not engine.finished and not scheduler.finished:
                if collab.is_aborted:
                    raise RuntimeError("async workflow aborted")
                dispatch()
                identity = scheduler.next_result()
                if scheduler.finished:
                    break
                if identity is not None:
                    call = completed.pop(identity, None)
                    if call is None:
                        raise ValueError("scheduler selected an incomplete result")
                    result = results.take(identity)
                    version = engine.version
                    published = engine.accept(call.assignment, self._normalize(result, call.assignment))
                    submitted.remove(identity)
                    scheduler.on_accept(call.assignment)
                    self._emit("accept", call.assignment, version, call.client_name)
                    if published:
                        dispatch()
                        self._hook("on_model", engine.model, engine.version)
                    continue
                # A simulated phase may unlock another call without accepting a
                # result yet. Process that work before waiting for physical I/O.
                dispatch()
                if not inflight:
                    raise RuntimeError("async workflow cannot make progress")
                try:
                    client_name, identity, result, error = outcomes.get(timeout=0.1)
                except queue.Empty:
                    continue
                call = inflight.pop(client_name)
                assignment = call.assignment
                if identity != assignment.assignment_id:
                    raise ValueError("result identity mismatch")
                if error is not None or result is None:
                    raise RuntimeError(f"assignment {assignment.assignment_id} failed: {error or 'no result'}")
                results.put(assignment.assignment_id, result)
                completed[assignment.assignment_id] = call
                scheduler.on_result(call)


class ResearchRecipe(CollabRecipe):
    """Example-level composition; reproducibility options never enter AsyncRecipe."""

    def __init__(
        self,
        job_name: str,
        client,
        application,
        model_adapter,
        num_updates: int,
        num_active_jobs: int,
        scheduler,
        train_kwargs=None,
        buffer_size: int = 1,
        min_open_slots: int = 1,
        server_lr: float = 1.0,
        weight_metric: str | None = None,
        update_transform=None,
        call_timeout: float = 3600.0,
        **collab_options,
    ):
        workflow = ResearchWorkflow(
            application,
            model_adapter,
            num_updates,
            num_active_jobs,
            scheduler,
            train_kwargs,
            buffer_size,
            min_open_slots,
            server_lr,
            weight_metric,
            update_transform,
            call_timeout,
        )
        collab_options.setdefault("min_clients", scheduler.required_workers(num_active_jobs))
        super().__init__(job_name=job_name, server=workflow, client=client, **collab_options)

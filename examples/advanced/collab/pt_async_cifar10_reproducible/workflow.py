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
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from pt_async_cifar10_reproducible.disk_storage import DiskStore, load_checkpoint

from nvflare.collab import collab
from nvflare.collab.async_engine import AsyncEngine
from nvflare.collab.async_recipe import AsyncWorkflow
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
        return load_checkpoint(self.base_path)


class ResearchEngine(AsyncEngine):
    """Keep replay reservations on disk; inherit all acceptance/aggregation math."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_dir = Path(collab.workspace.get_run_dir(collab.fl_ctx.get_job_id()))
        self.bases = DiskStore(self.run_dir / "assignment_bases")

    def reserve(self, logical_names, metadata=None):
        # ReplayScheduler validates the complete name/metadata sequences once.
        count = min(len(logical_names), self.available_slots)
        if not count:
            return []
        path = self.bases.path(self.version)
        if not path.exists():
            # Pickling the immutable current state needs no additional snapshot.
            self.bases.put(self.version, self.model)
        assignments = []
        for index in range(count):
            assignment = DiskAssignment(
                self._next_id,
                None,
                logical_names[index],
                self.version,
                str(path),
                deepcopy(metadata[index]) if metadata else {},
            )
            self._next_id += 1
            self.active[assignment.assignment_id] = assignment
            assignments.append(assignment)
        return assignments


class ResearchWorkflow(AsyncWorkflow):
    """Own fixed replay, optional train arguments, and retained disk results.

    Core supplies Collab calls, lifecycle and numerical acceptance. This example
    owns logical reservations, worker reuse and results held for replay.
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

    _engine_class = ResearchEngine

    def _run(self, engine, names, submit, outcomes):
        scheduler = self.scheduler
        try:
            results = DiskStore(engine.run_dir / "assignment_results")
            scheduler.initialize(engine, names)
            inflight, completed, submitted = {}, {}, set()

            def dispatch():
                if engine.finished:
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

            while not engine.finished:
                if collab.is_aborted:
                    raise RuntimeError("async workflow aborted")
                dispatch()
                identity = scheduler.next_result()
                if identity is not None:
                    call = completed.pop(identity, None)
                    if call is None:
                        raise ValueError("scheduler selected an incomplete result")
                    result = results.load(identity)
                    version = engine.version
                    published = engine.accept(call.assignment, self._normalize(result, call.assignment))
                    submitted.remove(identity)
                    scheduler.on_accept(call.assignment)
                    self._emit("accept", call.assignment, version, call.client_name)
                    if published:
                        dispatch()
                        self._hook("on_model", engine.model, engine.version)
                    continue
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
        finally:
            scheduler.close()


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
        collab_options.setdefault("min_clients", 1)
        super().__init__(job_name=job_name, server=workflow, client=client, **collab_options)

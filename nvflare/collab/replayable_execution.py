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

"""Shared controlled-execution boundary for Collab assignment workflows."""

import pickle
import queue
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from nvflare.apis.simulation_plan import ASYNC_EXECUTION, SIMULATION_PLAN_PROP, PlannedAsyncEvents


def _no_event(*args):
    pass


@dataclass(frozen=True)
class StoredAssignment:
    """Assignment whose immutable base is retained by the simulation runtime."""

    assignment_id: int
    client_name: str
    logical_name: str
    base_version: int
    base_path: str
    metadata: dict = field(default_factory=dict)

    @property
    def base_model(self):
        # The path is created by the server-side runtime, never supplied by a client.
        with open(self.base_path, "rb") as stream:
            return pickle.load(stream)


def _seed_record(assignment):
    return {"train_seed": assignment.metadata["train_seed"]} if "train_seed" in assignment.metadata else {}


@dataclass
class AsyncExecutionPort:
    """Algorithm operations needed by a Collab workflow's execution policy.

    ``state`` exposes model, version, active, available_slots, finished,
    reserve(names, metadata, factory), and accept(assignment, result).
    The callbacks keep algorithm-specific normalization and observers outside
    the simulator's assignment and result scheduling policy.
    """

    state: Any
    clients: list[str]
    contribution_budget: int
    submit: Callable
    receive: Callable
    live_run: Callable
    normalize: Callable
    aborted: Callable
    on_dispatch: Callable = _no_event
    on_accept: Callable = _no_event
    on_publish: Callable = _no_event
    on_time: Callable = _no_event
    make_assignment: Callable | None = None


def run_async_execution(port: AsyncExecutionPort, fl_ctx) -> None:
    """Use live arrivals or the simulator plan through the same algorithm port."""
    runtime = fl_ctx.get_prop(SIMULATION_PLAN_PROP) if fl_ctx is not None else None
    if runtime is None or not runtime.has_section(ASYNC_EXECUTION):
        port.live_run()
        return
    PlannedAsyncExecution(port, runtime).run()


class PlannedAsyncExecution:
    """Reserve logical work independently of physical sites; replay acceptance."""

    def __init__(self, port: AsyncExecutionPort, runtime):
        self.port = port
        self.state = port.state
        self.runtime = runtime

    def _make_assignment(self, identity, logical_name, version, model, metadata):
        path = self.runtime.save_model(version, model)
        if self.port.make_assignment is not None:
            return self.port.make_assignment(identity, logical_name, version, path, metadata)
        return StoredAssignment(identity, logical_name, logical_name, version, path, metadata)

    def run(self):
        plan = self.runtime.claim(ASYNC_EXECUTION)
        sequence = plan["logical_sequence"]
        seeds = plan.get("train_seeds")
        self.physical_sites = plan.get("physical_sites")
        total = self.port.contribution_budget
        if len(sequence) != total:
            raise ValueError(f"async plan has {len(sequence)} assignments; workflow requires {total}")
        if not self.port.clients:
            raise ValueError("planned async execution requires a physical client")
        if self.physical_sites is not None and set(self.physical_sites) - set(self.port.clients):
            raise ValueError("async plan names unavailable physical sites")

        if "timing" in plan:
            self._run_timed(sequence, seeds, plan["timing"])
            return
        order = plan["arrival_sequence"]

        available = list(self.port.clients)
        pending = deque()
        inflight = {}
        completed_sites = {}
        ready = set()
        next_reservation = 0
        next_acceptance = 0
        self.runtime.save_model(self.state.version, self.state.model)

        def dispatch():
            nonlocal next_reservation
            slots = self.state.available_slots
            logical_names = sequence[next_reservation : next_reservation + slots]
            metadata = [
                ({"train_seed": seeds[i]} if seeds is not None else {})
                for i in range(next_reservation, next_reservation + len(logical_names))
            ]
            assignments = self.state.reserve(logical_names, metadata, self._make_assignment)
            expected_ids = list(range(next_reservation, next_reservation + len(logical_names)))
            if [assignment.assignment_id for assignment in assignments] != expected_ids:
                raise ValueError("async state must reserve every planned assignment with sequential IDs")
            next_reservation += len(assignments)
            pending.extend(assignments)
            for assignment in assignments:
                self.runtime.trace(
                    "assignments.jsonl",
                    {
                        "assignment_id": assignment.assignment_id,
                        "logical_name": assignment.logical_name,
                        "base_version": assignment.base_version,
                        **_seed_record(assignment),
                    },
                )
            while pair := self._take_dispatch_pair(available, pending):
                site, assignment = pair
                kwargs = {"train_seeds": {site: assignment.metadata["train_seed"]}} if seeds is not None else {}
                self.port.submit(site, assignment, **kwargs)
                inflight[site] = assignment.assignment_id
                self.runtime.trace("dispatch.jsonl", {"assignment_id": assignment.assignment_id, "site": site})
                self.port.on_dispatch(assignment, self.state.version, site)

        while not self.state.finished:
            if self.port.aborted():
                raise RuntimeError("planned async workflow aborted")
            dispatch()
            expected = order[next_acceptance]
            if expected not in self.state.active:
                raise ValueError(f"replay assignment {expected} is incompatible with the available slots")
            if expected in ready:
                assignment = self.state.active[expected]
                result = self.runtime.load_result(expected)
                version = self.state.version
                published = self.state.accept(assignment, self.port.normalize(result, assignment))
                ready.remove(expected)
                next_acceptance += 1
                self.runtime.trace(
                    "accepted.jsonl",
                    {
                        "assignment_id": expected,
                        "logical_name": assignment.logical_name,
                        "base_version": assignment.base_version,
                        "current_version": version,
                        "published_version": self.state.version,
                        **_seed_record(assignment),
                    },
                )
                self.port.on_accept(assignment, version, completed_sites.pop(expected))
                if published:
                    self.runtime.save_model(self.state.version, self.state.model)
                    self.port.on_publish(self.state.model, self.state.version)
                continue
            if not inflight:
                raise RuntimeError("planned async workflow cannot make progress")
            try:
                site, identity, result, error = self.port.receive()
            except queue.Empty:
                continue
            if inflight.pop(site) != identity:
                raise ValueError("physical result identity mismatch")
            if error is not None or result is None:
                raise RuntimeError(f"assignment {identity} failed: {error or 'no result'}")
            self.runtime.save_result(identity, result)
            ready.add(identity)
            completed_sites[identity] = site
            available.append(site)

    def _run_timed(self, sequence, seeds, timing):
        """Apply logical events on a virtual clock; wall time only gates result readiness."""
        available = list(self.port.clients)
        pending = deque()
        inflight = {}
        completed_sites = {}
        ready = set()
        events = list(PlannedAsyncEvents(sequence, timing))
        self.runtime.save_model(self.state.version, self.state.model)

        def dispatch():
            while pair := self._take_dispatch_pair(available, pending):
                site, assignment = pair
                kwargs = {"train_seeds": {site: assignment.metadata["train_seed"]}} if seeds is not None else {}
                self.port.submit(site, assignment, **kwargs)
                inflight[site] = assignment.assignment_id
                self.runtime.trace("dispatch.jsonl", {"assignment_id": assignment.assignment_id, "site": site})
                self.port.on_dispatch(assignment, self.state.version, site)

        def collect(dispatch_pending=True):
            try:
                site, identity, result, error = self.port.receive()
            except queue.Empty:
                return
            if inflight.pop(site, None) != identity:
                raise ValueError("physical result identity mismatch")
            if error is not None or result is None:
                raise RuntimeError(f"assignment {identity} failed: {error or 'no result'}")
            self.runtime.save_result(identity, result)
            ready.add(identity)
            completed_sites[identity] = site
            available.append(site)
            if dispatch_pending:
                dispatch()

        index = 0
        while index < len(events) and not self.state.finished:
            if self.port.aborted():
                raise RuntimeError("planned async workflow aborted")
            instant, identity, phase = events[index]
            self.runtime.virtual_time = instant
            self.port.on_time(instant)
            if phase == "start":
                # Reserve all clients starting at the same virtual instant together,
                # so the refill threshold is checked once for the whole batch.
                starts = []
                while index < len(events) and events[index][0] == instant and events[index][2] == "start":
                    starts.append(events[index][1])
                    index += 1
                names = [sequence[i] for i in starts]
                metadata = [({"train_seed": seeds[i]} if seeds is not None else {}) for i in starts]
                assignments = self.state.reserve(names, metadata, self._make_assignment)
                if [assignment.assignment_id for assignment in assignments] != starts:
                    raise ValueError(f"logical starts at time {instant} require sequential available assignments")
                pending.extend(assignments)
                for assignment in assignments:
                    self.runtime.trace(
                        "assignments.jsonl",
                        {
                            "assignment_id": assignment.assignment_id,
                            "logical_name": assignment.logical_name,
                            "base_version": assignment.base_version,
                            "virtual_time": instant,
                            **_seed_record(assignment),
                        },
                    )
                    self.runtime.trace(
                        "phases.jsonl",
                        {"assignment_id": assignment.assignment_id, "phase": "start", "virtual_time": instant},
                    )
                dispatch()
                continue
            index += 1
            if identity not in self.state.active:
                raise ValueError(f"phase for unreserved assignment {identity}")
            self.runtime.trace("phases.jsonl", {"assignment_id": identity, "phase": phase, "virtual_time": instant})
            if phase != "upload":
                continue
            while identity not in ready:
                if self.port.aborted():
                    raise RuntimeError("planned async workflow aborted")
                if not inflight:
                    raise RuntimeError("planned async workflow cannot make progress")
                collect()
            assignment = self.state.active[identity]
            result = self.runtime.load_result(identity)
            version = self.state.version
            published = self.state.accept(assignment, self.port.normalize(result, assignment))
            ready.remove(identity)
            self.runtime.trace(
                "accepted.jsonl",
                {
                    "assignment_id": identity,
                    "logical_name": assignment.logical_name,
                    "base_version": assignment.base_version,
                    "current_version": version,
                    "published_version": self.state.version,
                    "virtual_time": instant,
                    **_seed_record(assignment),
                },
            )
            self.port.on_accept(assignment, version, completed_sites.pop(identity))
            if published:
                self.runtime.save_model(self.state.version, self.state.model)
                self.port.on_publish(self.state.model, self.state.version)

        if not self.state.finished:
            self.runtime.virtual_time = timing["horizon"]
            self.runtime.trace("phases.jsonl", {"phase": "horizon", "virtual_time": timing["horizon"]})

        # Retain results already in flight for inspection after a horizon stop.
        while inflight:
            if self.port.aborted():
                raise RuntimeError("planned async workflow aborted during drain")
            collect(dispatch_pending=False)

    def _take_dispatch_pair(self, available, pending):
        for site in available:
            for assignment in pending:
                if self.physical_sites is None or self.physical_sites[assignment.assignment_id] == site:
                    available.remove(site)
                    pending.remove(assignment)
                    return site, assignment
        return None

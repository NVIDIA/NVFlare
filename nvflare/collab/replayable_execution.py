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

import logging
import pickle
import queue
from collections import Counter, deque
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
class AsyncExecutionAdapter:
    """Algorithm operations needed by a Collab workflow's execution policy.

    ``state`` exposes model, version, active, available_slots, finished,
    reserve(names, metadata, factory), and accept(assignment, result).
    The callbacks keep algorithm-specific normalization and observers outside
    the simulator's assignment and result scheduling policy.

    ``make_planning_state`` builds a fresh model-free state with the same
    reservation and publication rules. Its ``accept`` receives a placeholder
    result and must not read model data or have external side effects.
    Fixed replay with preflight requires the workflow to finish on the last
    planned acceptance. Without preflight, the real state may finish early;
    successful replay drains started calls before releasing temporary bases.
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
    make_planning_state: Callable | None = None


def run_async_execution(adapter: AsyncExecutionAdapter, fl_ctx) -> None:
    """Use live arrivals or the simulator plan through the same algorithm adapter."""
    runtime = fl_ctx.get_prop(SIMULATION_PLAN_PROP) if fl_ctx is not None else None
    if runtime is None or not runtime.has_section(ASYNC_EXECUTION):
        adapter.live_run()
        return
    PlannedAsyncExecution(adapter, runtime).run()


class PlannedAsyncExecution:
    """Reserve logical work independently of physical sites; replay acceptance."""

    def __init__(self, adapter: AsyncExecutionAdapter, runtime):
        self.adapter = adapter
        self.state = adapter.state
        self.runtime = runtime
        self.expected = None
        self.retain_artifacts = True
        self._base_users = Counter()
        self._planned_acceptances = None

    def _make_assignment(self, identity, logical_name, version, model, metadata):
        path = self.runtime.save_model(version, model)
        if not self.retain_artifacts and self._planned_acceptances is None:
            self._base_users[version] += 1
        if self.adapter.make_assignment is not None:
            return self.adapter.make_assignment(identity, logical_name, version, path, metadata)
        return StoredAssignment(identity, logical_name, logical_name, version, path, metadata)

    def run(self):
        plan = self.runtime.claim(ASYNC_EXECUTION)
        sequence = plan["logical_sequence"]
        timing = plan.get("timing")
        self.physical_sites = plan.get("physical_sites")
        total = self.adapter.contribution_budget
        ordered = timing is not None and "events" in timing
        if not ordered and len(sequence) != total:
            raise ValueError(f"async plan has {len(sequence)} assignments; workflow requires {total}")
        if not self.adapter.clients:
            raise ValueError("planned async execution requires a physical client")
        if self.physical_sites is not None and set(self.physical_sites) - set(self.adapter.clients):
            raise ValueError("async plan names unavailable physical sites")

        if ordered and self.adapter.make_planning_state is None:
            raise ValueError("ordered timing requires an async planning state")
        events = list(PlannedAsyncEvents(sequence, timing)) if timing is not None else None
        preflight = None
        if self.adapter.make_planning_state is not None:
            preflight = _AsyncPreflight(self.adapter)
            execution = PlannedAsyncExecution(preflight.adapter, preflight)
            execution.physical_sites = self.physical_sites
            try:
                execution._run_plan(plan, events, strict_completion=True)
            except ValueError as error:
                raise ValueError(f"async plan preflight failed: {error}") from error
            for filename, record in preflight.records:
                self.runtime.trace("preflight.jsonl", {"trace": filename, **record})
            self.expected = iter(preflight.records)
            self._planned_acceptances = {
                record["assignment_id"] for filename, record in preflight.records if filename == "accepted.jsonl"
            }

        self.retain_artifacts = plan.get("retain_artifacts", True)
        if preflight is not None and not self.retain_artifacts:
            # Future reservations can reuse a base after its current assignments finish.
            self._base_users.update(
                record["base_version"] for filename, record in preflight.records if filename == "assignments.jsonl"
            )
        with self.runtime.artifacts(self.retain_artifacts):
            self._run_plan(plan, events, strict_completion=preflight is not None)
        if preflight is not None:
            if (missing := next(self.expected, None)) is not None:
                raise RuntimeError(f"async execution ended before preflight event: {missing}")
            self.runtime.verify_preflight(preflight.records)

    def _trace(self, filename, record):
        self.runtime.trace(filename, record)
        if self.expected is not None and filename != "dispatch.jsonl":
            expected = next(self.expected, None)
            if expected != (filename, record):
                raise RuntimeError(
                    f"async execution differs from preflight: expected {expected}, got {filename} {record}"
                )

    def _run_plan(self, plan, events, *, strict_completion):
        sequence = plan["logical_sequence"]
        seeds = plan.get("train_seeds")
        self.base_versions = plan.get("base_versions")
        if "timing" in plan:
            self._run_timed(sequence, seeds, events, plan["timing"])
            return
        order = plan["arrival_sequence"]

        available = list(self.adapter.clients)
        pending = deque()
        inflight = {}
        completed_sites = {}
        ready = set()
        next_reservation = 0
        next_acceptance = 0
        self._save_publication()

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
            self._verify_reservations(assignments, expected_ids)
            next_reservation += len(assignments)
            pending.extend(assignments)
            for assignment in assignments:
                self._trace(
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
                self.adapter.submit(site, assignment, **kwargs)
                inflight[site] = assignment.assignment_id
                self._trace("dispatch.jsonl", {"assignment_id": assignment.assignment_id, "site": site})
                self.adapter.on_dispatch(assignment, self.state.version, site)

        while not self.state.finished and next_acceptance < len(order):
            if self.adapter.aborted():
                raise RuntimeError("planned async workflow aborted")
            dispatch()
            expected = order[next_acceptance]
            if expected not in self.state.active:
                raise ValueError(f"replay assignment {expected} is incompatible with the available slots")
            if expected in ready:
                assignment = self.state.active[expected]
                result = self.runtime.load_result(expected)
                version = self.state.version
                published = self.state.accept(assignment, self.adapter.normalize(result, assignment))
                self._verify_publication(assignment, version, published)
                ready.remove(expected)
                next_acceptance += 1
                self._trace(
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
                self.adapter.on_accept(assignment, version, completed_sites.pop(expected))
                if published:
                    self._save_publication()
                    self.adapter.on_publish(self.state.model, self.state.version)
                self._discard_consumed_assignment(assignment)
                continue
            if not inflight:
                raise RuntimeError("planned async workflow cannot make progress")
            try:
                site, identity, result, error = self.adapter.receive()
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

        # A custom state can finish before every planned assignment is accepted.
        # Started calls still need their base files until their results arrive.
        while inflight:
            if self.adapter.aborted():
                raise RuntimeError("planned async workflow aborted during drain")
            try:
                site, identity, result, error = self.adapter.receive()
            except queue.Empty:
                continue
            if inflight.pop(site, None) != identity:
                raise ValueError("physical result identity mismatch")
            if error is not None or result is None:
                logging.getLogger(__name__).warning(
                    "Discarding failed assignment %s after fixed replay finished: %s",
                    identity,
                    error or "no result",
                )
            elif self.retain_artifacts:
                self.runtime.save_result(identity, result)

        if not self.retain_artifacts:
            for assignment in self.state.active.values():
                self._release_base(assignment)

        if not self.state.finished or (strict_completion and next_acceptance != len(order)):
            raise ValueError("fixed replay must finish after accepting its complete arrival sequence")

    def _verify_reservations(self, assignments, expected_ids):
        if [assignment.assignment_id for assignment in assignments] != expected_ids:
            raise ValueError(f"async state could not reserve the planned assignments {expected_ids}")
        for assignment in assignments:
            identity = assignment.assignment_id
            if self.base_versions is not None and assignment.base_version != self.base_versions[identity]:
                raise ValueError(
                    f"assignment {identity} has base version {assignment.base_version}; plan expects {self.base_versions[identity]}"
                )

    def _save_publication(self):
        if self.retain_artifacts:
            self.runtime.save_model(self.state.version, self.state.model)

    def _verify_publication(self, assignment, version, published):
        if self.adapter.make_planning_state is not None and bool(published) != (self.state.version > version):
            raise RuntimeError(
                f"async execution reported inconsistent publication at assignment {assignment.assignment_id}"
            )

    def _discard_consumed_assignment(self, assignment):
        if not self.retain_artifacts:
            self.runtime.discard_result(assignment.assignment_id)
            self._release_base(assignment)

    def _release_base(self, assignment):
        version = assignment.base_version
        self._base_users[version] -= 1
        if not self._base_users[version]:
            self.runtime.discard_model(version)
            del self._base_users[version]

    def _run_timed(self, sequence, seeds, events, timing):
        """Apply logical events on a virtual clock; wall time only gates result readiness."""
        available = list(self.adapter.clients)
        pending = deque()
        inflight = {}
        completed_sites = {}
        ready = set()
        ordered = "events" in timing
        horizon = timing["horizon"]
        released = set() if timing.get("release_on_train", False) else None
        reserve_options = {}
        if released is not None:
            reserve_options["released_assignments"] = released
        if timing.get("speculative", False):
            reserve_options["speculative"] = True
        self._save_publication()

        def dispatch():
            while pair := self._take_dispatch_pair(available, pending):
                site, assignment = pair
                if self.adapter.aborted():
                    raise RuntimeError("planned async workflow aborted")
                kwargs = {"train_seeds": {site: assignment.metadata["train_seed"]}} if seeds is not None else {}
                self.adapter.submit(site, assignment, **kwargs)
                inflight[site] = assignment.assignment_id
                self._trace("dispatch.jsonl", {"assignment_id": assignment.assignment_id, "site": site})
                self.adapter.on_dispatch(assignment, self.state.version, site)

        def collect(dispatch_pending=True):
            try:
                site, identity, result, error = self.adapter.receive()
            except queue.Empty:
                return
            if inflight.pop(site, None) != identity:
                raise ValueError("physical result identity mismatch")
            required = self._planned_acceptances is None or identity in self._planned_acceptances
            if error is not None or result is None:
                if not ordered or required:
                    raise RuntimeError(f"assignment {identity} failed: {error or 'no result'}")
                logging.getLogger(__name__).warning(
                    "Discarding failed assignment %s outside the planned acceptances: %s",
                    identity,
                    error or "no result",
                )
                result = None
            if dispatch_pending:
                if result is not None and (required or self.retain_artifacts):
                    self.runtime.save_result(identity, result)
                    completed_sites[identity] = site
                elif not self.retain_artifacts:
                    self._release_base(self.state.active[identity])
                # Even a discarded result can unblock the final phase at the horizon.
                ready.add(identity)
                available.append(site)
                dispatch()
            elif not self.retain_artifacts:
                self._release_base(self.state.active[identity])
            elif result is not None:
                self.runtime.save_result(identity, result)

        index = 0
        while index < len(events) and not self.state.finished:
            if self.adapter.aborted():
                raise RuntimeError("planned async workflow aborted")
            instant, identity, phase = events[index]
            if not ordered or phase != "upload":
                self.runtime.virtual_time = instant
                self.adapter.on_time(instant)
            if phase == "start":
                # Reserve all clients starting at the same virtual instant together,
                # so the refill threshold is checked once for the whole batch.
                starts = []
                while index < len(events) and events[index][0] == instant and events[index][2] == "start":
                    starts.append(events[index][1])
                    index += 1
                names = [sequence[i] for i in starts]
                metadata = [({"train_seed": seeds[i]} if seeds is not None else {}) for i in starts]
                assignments = self.state.reserve(names, metadata, self._make_assignment, **reserve_options)
                self._verify_reservations(assignments, starts)
                pending.extend(assignments)
                for assignment in assignments:
                    self._trace(
                        "assignments.jsonl",
                        {
                            "assignment_id": assignment.assignment_id,
                            "logical_name": assignment.logical_name,
                            "base_version": assignment.base_version,
                            "virtual_time": instant,
                            **_seed_record(assignment),
                        },
                    )
                    self._trace(
                        "phases.jsonl",
                        {"assignment_id": assignment.assignment_id, "phase": "start", "virtual_time": instant},
                    )
                dispatch()
                continue
            index += 1
            if identity not in self.state.active:
                raise ValueError(f"phase for unreserved assignment {identity}")
            while phase == "upload" and identity not in ready:
                if self.adapter.aborted():
                    raise RuntimeError("planned async workflow aborted")
                if not inflight:
                    raise RuntimeError("planned async workflow cannot make progress")
                collect()
            if self.adapter.aborted():
                raise RuntimeError("planned async workflow aborted")
            if ordered and phase == "upload":
                self.runtime.virtual_time = instant
                self.adapter.on_time(instant)
            self._trace("phases.jsonl", {"assignment_id": identity, "phase": phase, "virtual_time": instant})
            if ordered and instant >= horizon:
                break
            if phase != "upload":
                if phase == "train" and released is not None:
                    released.add(identity)
                continue
            assignment = self.state.active[identity]
            result = self.runtime.load_result(identity)
            version = self.state.version
            published = self.state.accept(assignment, self.adapter.normalize(result, assignment))
            self._verify_publication(assignment, version, published)
            if released is not None:
                released.discard(identity)
            ready.remove(identity)
            self._trace(
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
            self.adapter.on_accept(assignment, version, completed_sites.pop(identity))
            if published:
                self._save_publication()
                self.adapter.on_publish(self.state.model, self.state.version)
            self._discard_consumed_assignment(assignment)

        if ordered and index < len(events):
            raise ValueError("ordered plan has events after the workflow finished")
        if not self.state.finished:
            if ordered:
                if self.runtime.virtual_time < horizon:
                    raise ValueError("ordered plan ended before the workflow finished or reached its horizon")
            else:
                self.runtime.virtual_time = horizon
            self._trace("phases.jsonl", {"phase": "horizon", "virtual_time": self.runtime.virtual_time})

        if not self.retain_artifacts:
            for identity in completed_sites:
                self._discard_consumed_assignment(self.state.active[identity])
            for assignment in pending:
                self._release_base(assignment)

        # Drain active calls without admitting pending work after the stop.
        while inflight:
            if self.adapter.aborted():
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


class _AsyncPreflight:
    """Run the same scheduler with immediate placeholder results and no artifact I/O."""

    def __init__(self, adapter):
        self.records = []
        self.results = deque()
        self.adapter = AsyncExecutionAdapter(
            state=adapter.make_planning_state(),
            clients=adapter.clients,
            contribution_budget=adapter.contribution_budget,
            submit=self.submit,
            receive=self.results.popleft,
            live_run=_no_event,
            normalize=lambda result, assignment: result,
            aborted=adapter.aborted,
        )

    def submit(self, site, assignment, **kwargs):
        self.results.append((site, assignment.assignment_id, True, None))

    def trace(self, filename, record):
        if filename != "dispatch.jsonl":
            self.records.append((filename, record))

    def save_model(self, version, model):
        return ""

    def save_result(self, assignment_id, result):
        pass

    def load_result(self, assignment_id):
        return True

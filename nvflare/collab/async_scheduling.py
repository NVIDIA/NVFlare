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

"""Scheduling contracts and the default actual-arrival policy."""

from collections import deque
from dataclasses import dataclass
from typing import Protocol

from nvflare.collab.async_workflow import Assignment


@dataclass(frozen=True)
class ScheduledCall:
    """Bind a reserved logical assignment to a physical Collab worker."""

    assignment: Assignment
    client_name: str


@dataclass(frozen=True)
class AsyncEvent:
    """Lightweight trace entry: no model snapshots are retained by events."""

    phase: str
    assignment_id: int
    logical_name: str
    client_name: str
    base_version: int
    current_version: int
    metadata: dict
    time: float | None = None


class AsyncScheduler(Protocol):
    """Single-owner scheduler; only the workflow accepts results into the engine.

    initialize receives the engine and physical workers. dispatch reserves bases
    through engine.dispatch/reserve and binds pending jobs to idle workers.
    on_result records physical completion. next_result chooses a completed
    assignment ID for acceptance, or None to wait. on_accept observes logical retirement.
    finished can stop a run early. Model-update
    logic belongs to the engine/transform, not to the scheduler.
    """

    def required_workers(self, num_active_jobs) -> int:
        """Return the minimum physical workers required to start this policy."""

    def initialize(self, engine, client_names):
        """Initialize run-local scheduling state and the physical worker pool."""

    def dispatch(self) -> list[ScheduledCall]:
        """Reserve logical work and bind new calls to idle physical workers."""

    def on_result(self, call):
        """Observe physical completion after its raw result has been stored."""

    def next_result(self) -> int | None:
        """Select a completed result to accept, or None while waiting."""

    def on_accept(self, assignment):
        """Observe logical retirement after aggregation consumes a result."""

    @property
    def finished(self) -> bool:
        """Whether the policy has ended this run."""

    def close(self):
        """Release scheduling state, including after partial initialization."""


class ArrivalOrderScheduler:
    """Actual-arrival mode: one outstanding assignment per physical worker."""

    def __init__(self):
        self._available = []
        self._completed = deque()

    def required_workers(self, num_active_jobs):
        return num_active_jobs

    def initialize(self, engine, client_names):
        if len(client_names) < engine.num_active_jobs:
            raise ValueError("num_active_jobs exceeds available clients")
        self.engine = engine
        self._available = list(client_names)
        self._completed.clear()

    def dispatch(self):
        calls = []
        for assignment in self.engine.dispatch(self._available):
            self._available.remove(assignment.client_name)
            calls.append(ScheduledCall(assignment, assignment.client_name))
        return calls

    def on_result(self, call):
        self._completed.append(call.assignment.assignment_id)

    def next_result(self):
        return self._completed.popleft() if self._completed else None

    def on_accept(self, assignment):
        self._available.append(assignment.client_name)

    @property
    def finished(self):
        return False

    def close(self):
        self._available.clear()
        self._completed.clear()

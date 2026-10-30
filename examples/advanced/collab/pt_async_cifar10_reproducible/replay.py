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

"""Research-only fixed-order replay, independent of physical RPC timing."""

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScheduledCall:
    assignment: Any
    client_name: str


class ReplayScheduler:
    """Accept a predefined assignment-ID sequence regardless of RPC completion.

    Supply logical_sequence to reserve logical jobs independently of worker
    count (K may exceed physical workers). Its entries name each assignment in
    dispatch order, including repeated logical clients. Per-assignment metadata
    can carry training seeds or experimental controls to a train_kwargs builder.
    Replay fixes acceptance and assignment bases; it does not simulate time.
    """

    def __init__(
        self,
        arrival_sequence: list[int],
        logical_sequence: list[str],
        assignment_metadata: list[dict] | None = None,
    ):
        self.arrival_sequence = arrival_sequence
        self.logical_sequence = logical_sequence
        self.assignment_metadata = assignment_metadata
        self._available = []
        self._pending = deque()
        self._ready = set()
        self._cursor = 0

    def initialize(self, engine, client_names):
        total = engine.num_updates * engine.buffer_size
        if (
            len(self.arrival_sequence) != total
            or any(type(value) is not int for value in self.arrival_sequence)
            or set(self.arrival_sequence) != set(range(total))
        ):
            raise ValueError("arrival_sequence must permute the complete assignment-ID budget")
        if len(self.logical_sequence) != total or any(not isinstance(n, str) or not n for n in self.logical_sequence):
            raise ValueError("logical_sequence must name every assignment")
        if self.assignment_metadata is not None and (
            len(self.assignment_metadata) != total or any(not isinstance(m, dict) for m in self.assignment_metadata)
        ):
            raise ValueError("assignment_metadata must contain a dictionary for every assignment")
        if not client_names:
            raise ValueError("replay requires at least one physical worker")
        self.engine = engine
        self._available = list(client_names)
        self._cursor = 0
        self._reserved = 0
        self._ready.clear()
        self._pending.clear()

    def dispatch(self):
        end = self._reserved + self.engine.num_active_jobs
        metadata = None if self.assignment_metadata is None else self.assignment_metadata[self._reserved : end]
        assignments = self.engine.reserve(self.logical_sequence[self._reserved : end], metadata)
        self._reserved += len(assignments)
        self._pending.extend(assignments)
        calls = []
        while self._available and self._pending:
            calls.append(ScheduledCall(self._pending.popleft(), self._available.pop(0)))
        return calls

    def on_result(self, call):
        self._ready.add(call.assignment.assignment_id)
        self._available.append(call.client_name)

    def next_result(self):
        if self._cursor == len(self.arrival_sequence):
            return None
        expected = self.arrival_sequence[self._cursor]
        if expected not in self.engine.active:
            raise ValueError(f"replay assignment {expected} is not dispatched; sequence is incompatible with K/B/O")
        return expected if expected in self._ready else None

    def on_accept(self, assignment):
        self._ready.remove(assignment.assignment_id)
        self._cursor += 1

    def close(self):
        self._available.clear()
        self._pending.clear()
        self._ready.clear()
        self.engine = None

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

"""Seeded paper-experiment phases; independent of physical RPC timing."""

import heapq
import json
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Phase:
    finish_time: float
    phase: str
    assignment_id: int

    def __lt__(self, other):
        # Match the reference heap: ties compare times only, not assignment IDs.
        return self.finish_time < other.finish_time


class PhaseSchedule:
    """Release logical clients after training, and accept only due uploads.

    Runtime profiles persist by logical client. Keep RandomState, sampling order
    and the linear training-seed mapping from the reference experiment. Fixed
    replay and actual arrivals are different policies, not modes of core FedBuffRecipe.
    """

    def __init__(self, profiles, run_seed, max_time):
        self.profiles = profiles
        self.run_seed = run_seed
        self.max_time = max_time

    def initialize(self, coordinator, client_names):
        self.coordinator = coordinator
        self.rng = np.random.RandomState(self.run_seed)
        self.logical = sorted(self.profiles, key=lambda name: int(name.split("-", 1)[1]))
        self.physical = list(client_names)
        self.pending = deque()
        self.events = []
        self.ready = set()
        self.time = 0.0
        self.stopped = False
        self.reserve_due = True

    def dispatch(self):
        if self.stopped or self.coordinator.finished:
            return []
        if self.reserve_due:
            self.reserve_due = False
            count = min(self.coordinator.available_slots, len(self.logical))
            if count:
                indices = np.atleast_1d(self.rng.choice(len(self.logical), size=count, replace=False)).tolist()
                names = [self.logical[index] for index in indices]
                for index in sorted(indices, reverse=True):
                    del self.logical[index]
                for assignment in self.coordinator.reserve_for_phase(names, self.run_seed):
                    self.pending.append(assignment)
                    download = float(self.profiles[assignment.logical_name]["download_mean"])
                    heapq.heappush(self.events, Phase(self.time + download, "download", assignment.assignment_id))
        calls = []
        while self.physical and self.pending:
            calls.append((self.physical.pop(0), self.pending.popleft()))
        return calls

    def on_result(self, client_name, identity):
        self.physical.append(client_name)
        self.ready.add(identity)

    def next_result(self):
        while self.events:
            event = self.events[0]
            assignment = self.coordinator.active[event.assignment_id]
            profile = self.profiles[assignment.logical_name]
            if event.phase == "upload" and event.assignment_id not in self.ready:
                return None
            heapq.heappop(self.events)
            self.time = event.finish_time
            if event.phase == "download":
                duration = float(self.rng.exponential(float(profile["train_mean"])))
                heapq.heappush(self.events, Phase(self.time + duration, "train", event.assignment_id))
            elif event.phase == "train":
                self.logical.append(assignment.logical_name)
                self.logical.sort(key=lambda name: int(name.split("-", 1)[1]))
                mean = float(profile["upload_mean"])
                width = float(profile.get("upload_half_width", 0.02))
                duration = float(self.rng.uniform(max(0.0, mean - width), mean + width))
                heapq.heappush(self.events, Phase(self.time + duration, "upload", event.assignment_id))
            with (self.coordinator.run_dir / "phases.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"assignment_id": event.assignment_id, "phase": event.phase, "time": self.time}) + "\n"
                )
            if self.time >= self.max_time:
                self.stopped = True
                return None
            if event.phase == "upload":
                return event.assignment_id
        if not self.coordinator.finished:
            raise RuntimeError("No scheduled event remains")
        return None

    def on_accept(self, identity):
        self.ready.remove(identity)
        self.reserve_due = True

    def close(self):
        self.pending.clear()
        self.events.clear()
        self.ready.clear()
        self.coordinator = None

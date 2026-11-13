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

from collections import deque

from nvflare.apis.fl_context import FLContext
from nvflare.apis.simulation_plan import SIMULATION_PLAN_PROP, SimulationPlanRuntime
from nvflare.collab.replayable_execution import AsyncExecutionPort, run_async_execution


class _CounterState:
    """A small async algorithm that does not use the FedBuff coordinator."""

    def __init__(self):
        self.model = 0
        self.version = 0
        self.active = {}
        self.next_id = 0

    @property
    def available_slots(self):
        return min(2 - len(self.active), 2 - self.version - len(self.active))

    @property
    def finished(self):
        return self.version == 2

    def reserve(self, names, metadata, factory):
        assignments = []
        for name, details in zip(names[: self.available_slots], metadata):
            assignment = factory(self.next_id, name, self.version, self.model, details)
            self.active[self.next_id] = assignment
            assignments.append(assignment)
            self.next_id += 1
        return assignments

    def accept(self, assignment, result):
        assert assignment.base_model == 0
        self.model += result
        self.version += 1
        del self.active[assignment.assignment_id]
        return True


def test_generic_async_port_replays_non_fedbuff_algorithm(tmp_path):
    runtime = SimulationPlanRuntime(
        {"format_version": 1, "async_execution": {"logical_sequence": ["a", "b"], "arrival_sequence": [1, 0]}},
        tmp_path,
    )
    fl_ctx = FLContext()
    fl_ctx.set_prop(SIMULATION_PLAN_PROP, runtime)
    state = _CounterState()
    results = deque()
    accepted = []

    def submit(site, assignment, **kwargs):
        results.append((site, assignment.assignment_id, 1, None))

    port = AsyncExecutionPort(
        state=state,
        clients=["site-1", "site-2"],
        contribution_budget=2,
        submit=submit,
        receive=results.popleft,
        live_run=lambda: None,
        normalize=lambda result, assignment: result,
        aborted=lambda: False,
        on_accept=lambda assignment, version, site: accepted.append(assignment.assignment_id),
    )
    run_async_execution(port, fl_ctx)
    runtime.assert_consumed()
    assert accepted == [1, 0]
    assert state.model == 2


def test_generic_async_port_uses_live_path_without_plan():
    called = []
    port = AsyncExecutionPort(
        state=_CounterState(),
        clients=["site-1"],
        contribution_budget=2,
        submit=lambda *args, **kwargs: None,
        receive=lambda: None,
        live_run=lambda: called.append("live"),
        normalize=lambda result, assignment: result,
        aborted=lambda: False,
    )
    run_async_execution(port, FLContext())
    assert called == ["live"]

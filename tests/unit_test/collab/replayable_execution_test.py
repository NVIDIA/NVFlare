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

import json
from collections import deque

import pytest

from nvflare.apis.fl_context import FLContext
from nvflare.apis.simulation_plan import SIMULATION_PLAN_PROP, SimulationPlanRuntime
from nvflare.collab import FedBuffScheduleState
from nvflare.collab.replayable_execution import AsyncExecutionPort, run_async_execution


class _CounterState:
    """A small async algorithm that does not use the FedBuff coordinator."""

    def __init__(self, finish_version=2):
        self.model = 0
        self.version = 0
        self.active = {}
        self.next_id = 0
        self.finish_version = finish_version

    @property
    def available_slots(self):
        return min(2 - len(self.active), 2 - self.version - len(self.active))

    @property
    def finished(self):
        return self.version == self.finish_version

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


@pytest.mark.parametrize("retain_artifacts", [True, False])
@pytest.mark.parametrize("finish_version", [1, 2])
def test_generic_async_port_replays_non_fedbuff_algorithm(tmp_path, retain_artifacts, finish_version):
    runtime = SimulationPlanRuntime(
        {
            "format_version": 1,
            "async_execution": {
                "logical_sequence": ["a", "b"],
                "arrival_sequence": [1, 0],
                "retain_artifacts": retain_artifacts,
            },
        },
        tmp_path,
    )
    fl_ctx = FLContext()
    fl_ctx.set_prop(SIMULATION_PLAN_PROP, runtime)
    state = _CounterState(finish_version)
    results = deque()
    accepted = []

    def submit(site, assignment, **kwargs):
        results.append((site, assignment.assignment_id, 1, None))

    def on_accept(assignment, version, site):
        assert assignment.base_model == 0
        accepted.append(assignment.assignment_id)

    port = AsyncExecutionPort(
        state=state,
        clients=["site-1", "site-2"],
        contribution_budget=2,
        submit=submit,
        receive=results.popleft,
        live_run=lambda: None,
        normalize=lambda result, assignment: result,
        aborted=lambda: False,
        on_accept=on_accept,
    )
    run_async_execution(port, fl_ctx)
    runtime.assert_consumed()
    assert accepted == [1, 0][:finish_version]
    assert state.model == finish_version
    assert (tmp_path / "models").exists() == retain_artifacts
    assert not list(tmp_path.glob("work-*"))


@pytest.mark.parametrize("retain_artifacts", [True, False])
@pytest.mark.parametrize(
    ("result_order", "horizon_upload"),
    [
        pytest.param((1, 0), False, id="before-required-upload"),
        pytest.param((0, 1), False, id="during-drain"),
        pytest.param((1, 0), True, id="at-horizon"),
    ],
)
def test_ordered_replay_discards_failed_unaccepted_uploads(tmp_path, retain_artifacts, result_order, horizon_upload):
    steps = [(0, 0, "start"), (0, 1, "start"), (0, 0, "download"), (0, 0, "train"), (1, 0, "upload")]
    if horizon_upload:
        steps.extend([(1, 1, "download"), (1, 1, "train"), (2, 1, "upload")])
    runtime = SimulationPlanRuntime(
        {
            "format_version": 1,
            "async_execution": {
                "logical_sequence": ["a", "b"],
                "timing": {
                    "horizon": 2,
                    "speculative": True,
                    "events": [
                        {"virtual_time": time, "assignment_id": identity, "phase": phase}
                        for time, identity, phase in steps
                    ],
                },
                "retain_artifacts": retain_artifacts,
            },
        },
        tmp_path,
    )
    fl_ctx = FLContext()
    fl_ctx.set_prop(SIMULATION_PLAN_PROP, runtime)
    submissions = {}
    remaining = iter(result_order)

    def make_state():
        return FedBuffScheduleState(
            num_rounds=2 if horizon_upload else 1, concurrency=2, buffer_size=1, min_open_slots=1
        )

    def submit(site, assignment, **kwargs):
        submissions[assignment.assignment_id] = site

    def receive():
        identity = next(remaining)
        site = submissions[identity]
        return (site, identity, 1, None) if identity == 0 else (site, identity, None, "failed")

    def normalize(result, assignment):
        assert assignment.base_model is None  # The accepted assignment can still load its shared base.
        return result

    state = make_state()
    port = AsyncExecutionPort(
        state=state,
        clients=["site-1", "site-2"],
        contribution_budget=state.num_rounds,
        submit=submit,
        receive=receive,
        live_run=lambda: None,
        normalize=normalize,
        aborted=lambda: False,
        make_planning_state=make_state,
    )
    run_async_execution(port, fl_ctx)
    runtime.finalize()
    assert json.loads((tmp_path / "run.jsonl").read_text())["logical_trace_verified"] is True
    assert json.loads((tmp_path / "accepted.jsonl").read_text())["assignment_id"] == 0
    assert state.version == 1
    assert runtime.virtual_time == (2 if horizon_upload else 1)
    assert sorted(path.name for path in (tmp_path / "results").glob("*.pkl")) == (
        ["00000000.pkl"] if retain_artifacts else []
    )
    assert not list(tmp_path.glob("work-*"))


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

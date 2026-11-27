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

import hashlib
import json

import pytest

from nvflare.apis.simulation_plan import (
    ASYNC_EXECUTION,
    PlannedAsyncEvents,
    SimulationPlanRuntime,
    load_simulation_plan,
    validate_simulation_plan,
)


def test_plan_rejects_unsupported_sections():
    with pytest.raises(ValueError, match="unsupported simulation plan sections"):
        validate_simulation_plan({"format_version": 1, "unknown": {}})
    with pytest.raises(ValueError, match="unsupported simulation plan sections"):
        validate_simulation_plan({"format_version": 1, "round_participation": {"0": ["site-1"]}})


def test_async_plan_requires_complete_unique_arrival_order():
    plan = {"format_version": 1, "async_execution": {"logical_sequence": ["a", "b"], "arrival_sequence": [1, 0]}}
    assert validate_simulation_plan(plan) == plan
    plan["async_execution"]["arrival_sequence"] = [0, 0]
    with pytest.raises(ValueError, match="permute"):
        validate_simulation_plan(plan)
    plan["async_execution"]["arrival_sequence"] = [1, 0]
    plan["async_execution"]["physical_sites"] = ["site-1"]
    with pytest.raises(ValueError, match="one physical site per assignment"):
        validate_simulation_plan(plan)


def test_plan_runtime_requires_claim_and_finalizes_once(tmp_path):
    plan = {"format_version": 1, "async_execution": {"logical_sequence": ["a"], "arrival_sequence": [0]}}
    runtime = SimulationPlanRuntime(plan, tmp_path)
    with pytest.raises(RuntimeError, match="not consumed"):
        runtime.assert_consumed()
    assert runtime.claim(ASYNC_EXECUTION) == plan[ASYNC_EXECUTION]
    runtime.assert_consumed()
    with pytest.raises(RuntimeError, match="already claimed"):
        runtime.claim(ASYNC_EXECUTION)
    runtime.finalize()
    runtime.finalize()
    assert len((tmp_path / "run.jsonl").read_text().splitlines()) == 1
    assert json.loads((tmp_path / "run.jsonl").read_text())["logical_trace_verified"] is None


@pytest.mark.parametrize("change", [None, "missing", "extra", "record", "plan"])
def test_final_verification_reads_saved_plan_and_log(tmp_path, change):
    plan = {"format_version": 1, ASYNC_EXECUTION: {"logical_sequence": ["a"], "arrival_sequence": [0]}}
    plan_path = tmp_path / "simulation_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    runtime = SimulationPlanRuntime(load_simulation_plan(plan_path), tmp_path, plan_path=plan_path)
    record = {"assignment_id": 0, "logical_name": "a"}
    expected = [("assignments.jsonl", record)]
    if change != "missing":
        runtime.trace("assignments.jsonl", {**record, "logical_name": "b"} if change == "record" else record)
    if change == "extra":
        runtime.trace("accepted.jsonl", record)

    if change is None:
        runtime.verify_preflight(expected)
    else:
        if change == "plan":
            plan[ASYNC_EXECUTION]["logical_sequence"] = ["b"]
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
        error = {
            "missing": "differs from preflight at assignments.jsonl:1",
            "record": "differs from preflight at assignments.jsonl:1",
            "extra": "differs from preflight at accepted.jsonl:1",
            "plan": "saved simulation plan differs from the plan used at launch",
        }[change]
        with pytest.raises(RuntimeError, match=error):
            runtime.verify_preflight(expected)

    runtime.finalize()
    report = json.loads((tmp_path / "run.jsonl").read_text(encoding="utf-8"))
    canonical_plan = b'{"async_execution":{"arrival_sequence":[0],"logical_sequence":["a"]},"format_version":1}'
    assert report["plan_sha256"] == hashlib.sha256(canonical_plan).hexdigest()
    assert report["logical_trace_verified"] == (change is None)


def test_seeded_virtual_events_and_horizon_are_deterministic():
    delay = {"distribution": "exponential", "mean": 2.0}
    profile = {phase: delay for phase in ("download", "train", "upload")}
    plan = {
        "format_version": 1,
        "async_execution": {
            "logical_sequence": ["a", "b"],
            "timing": {"starts": [0, 1], "seed": 7, "profiles": {"a": profile, "b": profile}, "horizon": 10},
        },
    }
    timing = validate_simulation_plan(plan)["async_execution"]["timing"]
    first = list(PlannedAsyncEvents(["a", "b"], timing))
    assert first == list(PlannedAsyncEvents(["a", "b"], timing))
    assert first != list(PlannedAsyncEvents(["a", "b"], {**timing, "seed": 8}))
    assert [event for event in first if event[2] == "start"] == [(0.0, 0, "start"), (1.0, 1, "start")]
    assert all(event[0] < timing["horizon"] for event in first)
    plan["async_execution"]["timing"]["profiles"]["b"]["train"] = {"distribution": "uniform", "min": 2, "max": 1}
    with pytest.raises(ValueError, match="min exceeds max"):
        validate_simulation_plan(plan)
    plan["async_execution"]["timing"]["profiles"]["b"]["train"] = {"distribution": ["constant"], "value": 1}
    with pytest.raises(ValueError, match="unsupported delay"):
        validate_simulation_plan(plan)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("phase", "unexpected phase"),
        ("start", "ordered starts must reserve sequential assignment IDs"),
        ("horizon", "at most one final phase at or beyond the horizon"),
    ],
)
def test_ordered_plan_rejects_invalid_event_sequences(change, error):
    events = [
        {"virtual_time": 0, "assignment_id": 0, "phase": "start"},
        {"virtual_time": 0, "assignment_id": 1, "phase": "start"},
        {"virtual_time": 1, "assignment_id": 0, "phase": "download"},
        {"virtual_time": 1, "assignment_id": 0, "phase": "train"},
        {"virtual_time": 2, "assignment_id": 0, "phase": "upload"},
        {"virtual_time": 3, "assignment_id": 1, "phase": "download"},
    ]
    plan = {
        "format_version": 1,
        ASYNC_EXECUTION: {"logical_sequence": ["a", "b"], "timing": {"events": events, "horizon": 3}},
    }
    assert validate_simulation_plan(plan)[ASYNC_EXECUTION]["timing"]["events"][-1] == events[-1]
    if change == "phase":
        events[2]["phase"] = "train"
    elif change == "start":
        events[0]["assignment_id"] = 1
    else:
        events.append({"virtual_time": 3, "assignment_id": 1, "phase": "train"})
    with pytest.raises(ValueError, match=error):
        validate_simulation_plan(plan)

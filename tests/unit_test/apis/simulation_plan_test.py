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

import pytest

from nvflare.apis.simulation_plan import PlannedAsyncEvents, SimulationPlanRuntime, validate_simulation_plan


def test_round_plan_rejects_unknown_or_duplicate_participants():
    with pytest.raises(ValueError, match="unsupported simulation plan sections"):
        validate_simulation_plan({"format_version": 1, "unknown": {}})
    with pytest.raises(ValueError, match="duplicate client names"):
        validate_simulation_plan({"format_version": 1, "round_participation": {"0": ["site-1", "site-1"]}})


def test_round_plan_selects_exact_available_names_and_records_trace(tmp_path):
    runtime = SimulationPlanRuntime({"format_version": 1, "round_participation": {"0": ["site-3", "site-1"]}}, tmp_path)
    with pytest.raises(RuntimeError, match="not consumed"):
        runtime.assert_consumed()
    with pytest.raises(RuntimeError, match="unavailable"):
        runtime.select_round(0, ["site-1", "site-2"], 2)
    assert runtime.select_round(0, ["site-1", "site-2", "site-3"], 2) == ["site-3", "site-1"]
    with pytest.raises(RuntimeError, match="not fully accepted"):
        runtime.assert_consumed()
    runtime.validate_round_acceptance(0, ["site-1", "site-3"])
    runtime.assert_consumed()
    record = json.loads((tmp_path / "rounds.jsonl").read_text())
    assert record["selected"] == ["site-3", "site-1"]
    with pytest.raises(ValueError, match="expects 1"):
        runtime.select_round(0, ["site-1", "site-2", "site-3"], 1)


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


def test_round_acceptance_rejects_missing_contribution(tmp_path):
    runtime = SimulationPlanRuntime({"format_version": 1, "round_participation": {"0": ["a", "b"]}}, tmp_path)
    runtime.select_round(0, ["a", "b"], 2)
    with pytest.raises(RuntimeError, match="expected exactly"):
        runtime.validate_round_acceptance(0, ["a"])
    assert json.loads((tmp_path / "round_acceptance.jsonl").read_text())["accepted"] == ["a"]
    with pytest.raises(RuntimeError, match="not fully accepted"):
        runtime.assert_consumed()
    runtime.finalize()
    runtime.finalize()
    assert len((tmp_path / "run.jsonl").read_text().splitlines()) == 1


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

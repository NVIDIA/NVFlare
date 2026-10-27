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

"""Results v1.1: aggregates with seen/held-out, tags, 1.0 read-compat (spec §8)."""
from __future__ import annotations

from decent_vla.core.results import (
    SCHEMA_VERSION,
    compute_aggregates,
    read_results,
    tag_suites,
)


def _suites():
    return {
        "spatial": {"per_task": [
            {"task_id": 0, "name": "a", "n_episodes": 10, "n_success": 9,
             "success_rate": 0.9},
            {"task_id": 5, "name": "b", "n_episodes": 10, "n_success": 10,
             "success_rate": 1.0},
        ], "n_episodes": 20, "n_success": 19, "success_rate": 0.95},
    }


def test_tag_and_aggregate():
    suites = tag_suites(_suites(), heldout_eval_ids={"spatial": {5}})
    assert suites["spatial"]["per_task"][0]["tags"] == ["seen"]
    assert suites["spatial"]["per_task"][1]["tags"] == ["held_out"]
    agg = compute_aggregates(suites)
    assert agg["overall"] == {"n_episodes": 20, "n_success": 19, "success_rate": 0.95}
    assert agg["seen"]["n_tasks"] == 1 and agg["seen"]["success_rate"] == 0.9
    assert agg["held_out"]["n_tasks"] == 1 and agg["held_out"]["success_rate"] == 1.0


def test_aggregates_never_average_of_averages():
    suites = {"s1": {"per_task": [{"task_id": 0, "name": "x", "n_episodes": 90,
                                   "n_success": 0, "success_rate": 0.0}]},
              "s2": {"per_task": [{"task_id": 0, "name": "y", "n_episodes": 10,
                                   "n_success": 10, "success_rate": 1.0}]}}
    agg = compute_aggregates(tag_suites(suites, {}))
    assert agg["overall"]["success_rate"] == 0.1        # 10/100, not 0.5


def test_schema_11_and_10_read_compat(tmp_path):
    # 1.2 added the optional `provenance` block (see test_eval_provenance.py).
    assert SCHEMA_VERSION == "1.2"
    import json
    legacy = {"schema_version": "1.0", "run_id": "r", "backend": "mock:mock",
              "checkpoint": None, "benchmark": "mock", "config_hash": "sha256:x",
              "git_commit": None, "started_at": "t", "finished_at": "t",
              "wall_time_s": 1.0, "device": {"type": "cpu", "name": "cpu", "count": 1},
              "n_episodes_per_task": 1, "seed": 0, "suites": {}, "overall": {},
              "extra": {}}
    p = tmp_path / "r.json"
    p.write_text(json.dumps(legacy))
    rr = read_results(p)
    assert rr.aggregates == {} and rr.partition_manifest_hash is None

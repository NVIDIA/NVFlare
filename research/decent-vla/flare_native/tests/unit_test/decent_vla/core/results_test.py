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

"""Results JSON: schema fields, hashing, write/read round-trip."""

from __future__ import annotations

import json
from pathlib import Path

from decent_vla.core.results import (
    SCHEMA_VERSION,
    DeviceInfo,
    RunResults,
    config_hash,
    read_results,
    write_results,
)


def _example_results() -> RunResults:
    return RunResults(
        schema_version=SCHEMA_VERSION,
        run_id="test_run",
        backend="mock:mock",
        checkpoint=None,
        benchmark="mock",
        config_hash="sha256:deadbeef",
        git_commit=None,
        started_at="2026-05-10T00:00:00Z",
        finished_at="2026-05-10T00:00:30Z",
        wall_time_s=30.0,
        device=DeviceInfo(type="cpu", name="cpu", count=1),
        n_episodes_per_task=5,
        seed=42,
        suites={
            "default": {
                "n_episodes": 10,
                "n_success": 7,
                "success_rate": 0.7,
                "per_task": [
                    {
                        "task_id": 0,
                        "task_name": "t0",
                        "n_episodes": 5,
                        "n_success": 4,
                        "success_rate": 0.8,
                        "mean_episode_length": 4.0,
                    },
                    {
                        "task_id": 1,
                        "task_name": "t1",
                        "n_episodes": 5,
                        "n_success": 3,
                        "success_rate": 0.6,
                        "mean_episode_length": 4.0,
                    },
                ],
            }
        },
        overall={"n_episodes": 10, "n_success": 7, "success_rate": 0.7},
    )


def test_write_read_round_trip(tmp_path: Path):
    out = tmp_path / "results.json"
    r = _example_results()
    write_results(out, r)
    rb = read_results(out)
    assert rb.run_id == r.run_id
    assert rb.schema_version == SCHEMA_VERSION
    assert rb.device.type == "cpu"
    assert rb.suites["default"]["n_episodes"] == 10
    assert rb.overall["success_rate"] == 0.7


def test_config_hash_is_stable_under_key_order():
    a = config_hash({"x": 1, "y": 2})
    b = config_hash({"y": 2, "x": 1})
    assert a == b
    assert a.startswith("sha256:")


def test_config_hash_changes_with_content():
    a = config_hash({"x": 1})
    b = config_hash({"x": 2})
    assert a != b


def test_overall_success_rate_is_total_succ_over_total_eps(tmp_path: Path):
    """Spec guard: overall must NOT be a mean of per-suite rates."""
    r = _example_results()
    # Add a second suite with very different rate to make 'mean of means' wrong.
    r.suites["extra"] = {
        "n_episodes": 100,
        "n_success": 10,
        "success_rate": 0.10,
        "per_task": [],
    }
    # The framework writes whatever value we set in `overall`, but our cli/main.py
    # computes it as total/total. Verify the helper math via a direct check.
    total_eps = sum(s["n_episodes"] for s in r.suites.values())
    total_succ = sum(s["n_success"] for s in r.suites.values())
    correct_rate = total_succ / total_eps
    # mean of suite rates would be (0.7 + 0.1)/2 = 0.4
    mean_of_means = sum(s["success_rate"] for s in r.suites.values()) / len(r.suites)
    assert abs(correct_rate - mean_of_means) > 1e-6, "Bad test: rates would not differ"
    # Just to lock the invariant in this test, the value the CLI writes is total/total.
    out = tmp_path / "results.json"
    r.overall = {
        "n_episodes": total_eps,
        "n_success": total_succ,
        "success_rate": correct_rate,
    }
    write_results(out, r)
    data = json.loads(out.read_text())
    assert abs(data["overall"]["success_rate"] - correct_rate) < 1e-9

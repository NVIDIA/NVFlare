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

"""Public task rollup math is independent of startup inventory and wall time."""

from copy import deepcopy

import pytest

from nvflare.private.fed.resource_stats.contract import (
    SCHEMA_VERSION,
    ContractError,
    aggregate_cpu_consumed,
    aggregate_message_traffic,
    aggregate_resource_time,
    capacity_to_resource_time,
    derive_job_totals,
    normalize_measurement,
    validate_record,
)
from nvflare.tool.resources.resources_output import render_job_resources


def _capacity(gpu_count):
    return {"cpu": {"units": 2}, "memory": {"bytes": 1024}, "gpu": {"count": gpu_count}}


def _participant():
    return {
        "participant_name": "site-1",
        "role": "client",
        "received_at": "2026-10-29T12:00:00Z",
        "duration_seconds": 60,
        "resource_capacity": _capacity(0),
        "resource_time": aggregate_resource_time(
            [capacity_to_resource_time(_capacity(count), seconds) for count, seconds in [(0, 60), (1, 10), (1, 5)]]
        ),
        "resource_utilization": {
            "cpu_consumed": aggregate_cpu_consumed([{"cpu_seconds": amount} for amount in (2, 8, 4)]),
            "retained_content": {"bytes": 0},
            "message_traffic": aggregate_message_traffic(
                [{"sent_bytes": 1, "received_bytes": 2}, {"sent_bytes": 3, "received_bytes": 4}]
            ),
        },
        "issues": [],
    }


def test_task_capacity_time_and_cpu_are_authoritative_not_startup_products():
    participant = _participant()
    totals = derive_job_totals([participant])
    assert totals["resource_time"]["gpu"] == {"instance_seconds": 15}
    assert totals["resource_time"]["cpu"] == {"unit_seconds": 150}
    assert totals["resource_utilization"]["cpu_consumed"] == {"cpu_seconds": 14}
    assert totals["resource_utilization"]["message_traffic"] == {"sent_bytes": 4, "received_bytes": 6}
    assert participant["resource_capacity"]["gpu"] == {"count": 0}
    assert participant["duration_seconds"] == 60


def test_absent_containing_duration_retains_explicit_work_and_time():
    participant = _participant()
    original = derive_job_totals([participant])
    del participant["duration_seconds"]
    assert derive_job_totals([participant]) == original
    output = render_job_resources({"job_id": "job-1", "participants": [participant]})
    usage = output.split("Recorded resource utilization per participant", 1)[1]
    row = next(line for line in usage.splitlines() if line.startswith("site-1"))
    assert row.split()[1] == "—"
    assert "measurement duration unavailable" in output


def test_missing_explicit_time_is_invalid_not_reconstructed():
    participant = _participant()
    del participant["resource_time"]
    with pytest.raises(ContractError, match="resource_time"):
        derive_job_totals([participant])


def test_worker_gpu_time_and_selected_site_totals_remain_visible():
    participant = _participant()
    participant["resource_time"]["gpu"]["instance_seconds"] += 3600
    other = deepcopy(participant)
    other["participant_name"] = "site-2"
    other["resource_time"]["gpu"]["instance_seconds"] = 7200
    detail = deepcopy(participant)
    detail["workspace_filesystem"] = {"capacity_bytes": 1024}
    detail["issues"] = [issue for issue in detail["issues"] if issue["metric"] != "workspace_filesystem"] + []
    output = render_job_resources({"job_id": "job-1", "participants": [participant, other]}, detail)
    assert "Recorded capacity-time totals" in output
    assert "GPUs available 1 GPU-h" in output
    assert "Job-process startup capacity for site-1" in output
    assert "RAM: <0.01 GiB" in output
    assert "GPU: 0 GPUs" in output
    assert "AVG CPU USED" in output
    assert "0.23" in output


@pytest.mark.parametrize("field", ["resource_time", "cpu_consumed", "message_traffic"])
def test_partial_worker_does_not_erase_valid_subtotals(field):
    values = {
        "resource_time": {"gpu": {"instance_seconds": 0}, "cpu": {"unit_seconds": 2}, "memory": {"byte_seconds": 4}},
        "cpu_consumed": {"cpu_seconds": 2},
        "message_traffic": {"sent_bytes": 2, "received_bytes": 4},
    }
    aggregate = {
        "resource_time": aggregate_resource_time,
        "cpu_consumed": aggregate_cpu_consumed,
        "message_traffic": aggregate_message_traffic,
    }[field]
    result = aggregate([values[field], {"status": "unavailable"}])
    assert result["status"] == "partial"
    assert "issues" not in result
    assert {key: value for key, value in result.items() if key not in {"status", "issues"}} == values[field]
    normalize_measurement(field, result)


def test_public_report_requires_time_but_has_no_worker_inventory():
    record = _participant()
    record.pop("role")
    record.pop("received_at")
    record.update(
        schema_version=SCHEMA_VERSION,
        kind="nvflare.resource_stats.participant_summary",
        job_id="job-1",
        reported_at="2026-10-29T12:00:00Z",
        workspace_filesystem={"capacity_bytes": 1024},
    )
    validate_record(record)
    record["workers"] = []
    with pytest.raises(ContractError, match="workers"):
        validate_record(record)


def test_product_overflow_keeps_raw_capacity_and_other_time_dimensions():
    capacity = _capacity(1)
    capacity["memory"]["bytes"] = 10**308
    issues = []
    elapsed = capacity_to_resource_time(capacity, 2.0, scope_issues=issues)
    assert elapsed["status"] == "partial"
    assert issues == [{"metric": "resource_time", "code": "observation_incomplete"}]
    assert "issues" not in elapsed
    assert "memory" not in elapsed
    assert elapsed["cpu"] == {"unit_seconds": 4}
    assert elapsed["gpu"] == {"instance_seconds": 2}
    assert capacity["memory"]["bytes"] == 10**308

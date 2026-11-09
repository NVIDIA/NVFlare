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


"""Numeric arithmetic preserves independent values, zeros, and coverage."""

import pytest

from nvflare.private.fed.resource_stats.measurements import (
    aggregate_cpu_consumed,
    aggregate_message_traffic,
    aggregate_resource_time,
    aggregate_scope_issues,
    capacity_to_resource_time,
    scoped_measurement,
)


@pytest.mark.parametrize("memory_bytes", [0, 1024])
def test_capacity_time_uses_one_duration_and_preserves_zero_capacity(memory_bytes):
    capacity = {"cpu": {"units": 1.5}, "memory": {"bytes": memory_bytes}, "gpu": {"count": 0}}
    assert capacity_to_resource_time(capacity, 0.5) == {
        "cpu": {"unit_seconds": 0.75},
        "memory": {"byte_seconds": memory_bytes * 0.5},
        "gpu": {"instance_seconds": 0},
    }
    assert capacity_to_resource_time(capacity, 0) == {
        "cpu": {"unit_seconds": 0},
        "memory": {"byte_seconds": 0},
        "gpu": {"instance_seconds": 0},
    }
    assert capacity["cpu"] == {"units": 1.5}


def test_product_overflow_preserves_other_capacity_time_dimensions():
    capacity = {"cpu": {"units": 1e308}, "memory": {"bytes": 1024}, "gpu": {"count": 2}}
    issues = []
    assert capacity_to_resource_time(capacity, 2.0, issues) == {
        "status": "partial",
        "memory": {"byte_seconds": 2048},
        "gpu": {"instance_seconds": 4},
    }
    assert issues == [{"metric": "resource_time", "code": "observation_incomplete"}]


def test_aggregation_retains_missing_coverage_and_exact_integer_bytes():
    missing = {"status": "unavailable"}
    assert aggregate_cpu_consumed([{"cpu_seconds": 2.5}, missing]) == {
        "status": "partial",
        "cpu_seconds": 2.5,
    }
    assert aggregate_message_traffic(
        [{"sent_bytes": 10**50, "received_bytes": 7}, {"sent_bytes": 2, "received_bytes": 4}]
    ) == {"sent_bytes": 10**50 + 2, "received_bytes": 11}
    capacity = {"cpu": {"units": 2}, "memory": {"bytes": 8}, "gpu": {"count": 1}}
    values = [capacity_to_resource_time(capacity, duration) for duration in (2, 3)]
    assert aggregate_resource_time(values) == {
        "cpu": {"unit_seconds": 10},
        "memory": {"byte_seconds": 40},
        "gpu": {"instance_seconds": 5},
    }
    with pytest.raises(ValueError):
        aggregate_cpu_consumed([{"cpu_seconds": 1e308}, {"cpu_seconds": 1e308}])


def test_scope_assembly_moves_private_codes_once_without_mutating_the_producer():
    private = {"status": "partial", "cpu_seconds": 0.5, "issues": ["attribution_incomplete"]}
    issues = []
    assert scoped_measurement("cpu_consumed", private, issues) == {"status": "partial", "cpu_seconds": 0.5}
    assert issues == [{"metric": "cpu_consumed", "code": "attribution_incomplete"}]
    assert private["issues"] == ["attribution_incomplete"]


def test_capacity_time_shares_causes_and_never_invents_zero_when_duration_is_missing():
    capacity = {"status": "partial", "cpu": {"units": 2}}
    issues = [{"metric": "resource_capacity", "code": "attribution_incomplete"}]
    assert capacity_to_resource_time(capacity, 3, scope_issues=issues) == {
        "status": "partial",
        "cpu": {"unit_seconds": 6},
    }
    assert issues[-1] == {"metric": "resource_time", "code": "attribution_incomplete"}
    issues = []
    assert capacity_to_resource_time(capacity, None, issues) == {"status": "unavailable"}
    assert issues == [{"metric": "resource_time", "code": "observation_incomplete"}]


def test_aggregate_scope_issues_map_source_failures_and_retain_attribution():
    assert aggregate_scope_issues(
        [
            {"metric": "cpu_consumed", "code": "unsupported"},
            {"metric": "cpu_consumed", "code": "attribution_incomplete"},
            {"metric": "message_traffic", "code": "malformed_source"},
        ],
        {
            "cpu_consumed": {"status": "partial", "cpu_seconds": 1},
            "message_traffic": {"status": "partial", "sent_bytes": 2, "received_bytes": 3},
        },
    ) == [
        {"metric": "cpu_consumed", "code": "attribution_incomplete"},
        {"metric": "cpu_consumed", "code": "observation_incomplete"},
        {"metric": "message_traffic", "code": "observation_incomplete"},
    ]
